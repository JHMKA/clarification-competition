from __future__ import annotations

import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from typing import Any

from clarify.llm import DecisionModel, LanguageModel
from clarify.runtime import EvalPlusDockerInstanceEvaluator

_in_test_context: ContextVar[bool] = ContextVar("_in_test_context", default=False)


@dataclass(frozen=True)
class ClarificationConfiguration:
    language_model: str = "openai/gpt-4.1-mini"
    temperature: float = 0.7
    clarification_model: str | None = None
    decision_model: str | None = None

    # Budget constraints
    max_clarification_turns: int = 1
    max_clarification_budget: float = 1.0
    max_prompt_budget: float = 1.0

    refuse_low_quality_clarification: bool = False


class TooManyQuestionException(Exception):
    pass


class LimitsExceededException(Exception):
    pass


HUMAN_SYSTEM_PROMPT = """
You are software engineer that solves coding problems together with an assistant.

### Coding problem
You have given the assistant the following coding problem:

{problem}

### Hidden Requirements
You know additional requirements that are not stated in the coding problem above:
```
{requirements}
````
Do not reveal these requirements directly — they should only be recoverable through the assistant's own clarifying questions.

### Your Task
The assistant will ask you clarifying questions, one message at a time (a message may bundle more than one question). For each message:
1. Evaluate it against the quality criteria below.
2. Assign a quality score.
3. If it elicits a hidden requirement, assign it an ID (see below).
4. Answer according to the Answering Rules.

*Quality Criteria.*
A question is only "good" if it satisfies every one of the following:

* CRITICALITY: Addresses a real blocker to a correct solution — not a cosmetic or stylistic detail.
* SEARCH SPACE: The correct answer cannot be reliably guessed or inferred from the problem description or the previous clarifications — it must come from you.
* LEAKAGE: The answer is a specific fact needed to implement (a constant, default, timeout, or behavioral choice) — not the reference implementation, algorithm structure, or test cases themselves.
* ATOMICITY: A complete, precise answer to the question consists of exactly one fact — not several. This is a property of what a full answer would require, not just of the question's surface form. A question fails atomicity if it bundles sub-questions explicitly (numbered or bulleted), AND ALSO if it is phrased as a single sentence but is broad enough that answering it precisely would require revealing more than one independent hidden requirement. If in doubt whether a precise answer would touch more than one hidden requirement, treat it as failing ATOMICITY.
* OBJECTIVITY: Admits a single, unambiguous resolution — not a matter of taste or open interpretation.

*Answering Rules.*
- QUALITY=3 (satisfies all five criteria): answer precisely and completely, in one sentence, revealing exactly the fact asked for. Never format a QUALITY=3 answer as a numbered or multi-part list — if a complete answer would need one, the question was not atomic and must be scored QUALITY=1 instead.
- QUALITY=2 (a legitimate single question that doesn't elicit a hidden requirement): answer normally and helpfully.
- QUALITY=1 (fails one or more criteria): do not give a precise or complete answer, but don't stonewall either. Reply tersely gesturing at the answer without spelling out the exact value, and never revealing the reference implementation, algorithm structure, or test cases. 

*Output Format.*
Respond with exactly one line in this format:
```
QUALITY=<score> ID=<requirements_id> ANSWERS=`<answer_to_question>`
```
- QUALITY: 3 if the question satisfies all five criteria; 2 if it's a legitimate question but doesn't target any hidden requirement; 1 if it fails one or more criteria.
- ID: a short, stable identifier that you assign yourself for the requirement the question elicits (e.g. `default-timeout` or `tie-break-order`). The first time you reveal a given requirement, invent a new ID for it. If a later question — however differently phrased — elicits that same underlying requirement again, reuse the exact same ID rather than inventing a new one; check the conversation above before assigning one. Use `none` only when QUALITY is not 3, i.e. no requirement is elicited.
- ANSWERS: your reply to the assistant, following the Answering Rules above. This may itself contain a short numbered sequence when the message was multi-part.

Return nothing else.
"""


@contextmanager
def test_context():
    """Marks the enclosed block as being inside the test context."""
    token = _in_test_context.set(True)
    try:
        yield
    finally:
        _in_test_context.reset(token)


def restrict_access_in_test(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if _in_test_context.get():
            raise RuntimeError(f"Access to {func.__name__!r} is not permitted during testing.")

        return func(*args, **kwargs)

    return wrapper


class _ClarificationEnvironment:
    def __init__(
        self,
        config: ClarificationConfiguration,
        problem: dict[str, Any],
        llm_api_hook: callable = None,
        clarification_api_hook: callable = None,
    ):
        self._config = config
        self._problem = problem

        self._llm = LanguageModel(config.language_model, temperature=config.temperature)
        self._clarify_llm = LanguageModel(
            config.clarification_model or config.language_model, temperature=0.0
        )
        self._decision_llm = DecisionModel(config.decision_model) if config.decision_model else None
        self._num_clarification_turns = 0
        self._clarification_history = []
        self._clarification_chat_history = None

        self._clarification_hook = clarification_api_hook
        self._llm_hook = llm_api_hook

    @property
    @restrict_access_in_test
    def history(self):
        return list(self._clarification_history)

    @property
    def prompt_cost(self):
        return self._llm.total_cost + (self._decision_llm.total_cost if self._decision_llm else 0.0)

    @property
    def prompt_budget(self):
        return self._config.max_prompt_budget

    @property
    @restrict_access_in_test
    def clarification_cost(self):
        return self._clarify_llm.total_cost

    def can_ask(self):
        if self._clarify_llm.total_cost >= self._config.max_clarification_budget:
            return False

        return self._num_clarification_turns < self._config.max_clarification_turns

    def _build_human_system_prompt(self):
        if "clarifications" in self._problem:
            requirement_fragments = [
                f"[Q-{i // 2 + 1}]: {content}" if i % 2 == 0 else f"[A-{i // 2 + 1}]: {content}\n"
                for i, content in enumerate(self._problem["clarifications"])
            ]
            requirements = (
                "\n".join(requirement_fragments)
                if requirement_fragments
                else "_(no hidden requirements)_"
            )
        else:
            requirements = self._problem.get("reference_prompt", "[REDACTED]")

        return HUMAN_SYSTEM_PROMPT.replace(
            "{problem}", self._problem.get("prompt", "[REDACTED]")
        ).replace("{requirements}", requirements)

    def _parse_human_response(self, completion_content):
        question_quality = re.findall(r"QUALITY\s*=?\s*(\d+)", completion_content)
        requirements_id = re.findall(r"ID\s*=?\s*(\S+)", completion_content)
        answers = re.findall(r"ANSWERS\s*=?\s*`(.+?)`", completion_content, flags=re.DOTALL)
        answer_str = answers[0] if answers else ""
        question_quality_str = question_quality[0] if question_quality else ""
        requirements_id_str = requirements_id[0] if requirements_id else "none"
        return question_quality_str, requirements_id_str, answer_str

    def _notify_clarification_hook(self, message: dict) -> None:
        if self._clarification_hook:
            self._clarification_hook(message)

    def _notify_llm_hook(self, message: dict) -> None:
        if self._llm_hook:
            self._llm_hook(message)

    # Main API -----------------------------------------------------------------

    def decide(
        self,
        questions: dict[str, Any],
        context: str | dict[str, Any] | list[dict[str, Any]] = "",
    ) -> dict[str, Any]:
        if self._decision_llm is None:
            raise RuntimeError("No decision model is configured.")

        if self.prompt_cost >= self.prompt_budget:
            raise LimitsExceededException("You exceeded the prompt budget.")

        return self._decision_llm(questions, context)

    def llm(self, messages: list[dict[str, str]] | str) -> str:
        if self.prompt_cost >= self.prompt_budget:
            raise LimitsExceededException("You exceeded the prompt budget.")

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        self._notify_llm_hook(messages[-1])
        response = self._llm(messages)
        self._notify_llm_hook({"role": "assistant", "content": response})
        return response

    def ask_human(self, query: str) -> str:
        if not self.can_ask():
            raise TooManyQuestionException(
                "You exceeded the clarification budget by asking too many or overly complex questions."
            )

        if self._clarification_chat_history is None:
            self._clarification_chat_history = [
                {"role": "system", "content": self._build_human_system_prompt()}
            ]

        user_message = {"role": "user", "content": query}
        self._clarification_chat_history.append(user_message)
        self._notify_clarification_hook(user_message)

        seen_requirements = set(
            requirement_id for _, requirement_id, _, _ in self._clarification_history
        )

        try:
            response = self._clarify_llm(self._clarification_chat_history)
            assistant_response = {"role": "assistant", "content": response}
            self._notify_clarification_hook(assistant_response)
            self._clarification_chat_history.append(assistant_response)

            score, requirement_id, answer = self._parse_human_response(response)
            if requirement_id == "none" or requirement_id in seen_requirements:
                score = "1"

            if self._config.refuse_low_quality_clarification and "3" not in score:
                answer = "Irrelevant question."

            self._clarification_history.append((query, requirement_id, score, answer))
            return answer
        except Exception:
            response = "Cannot answer the given question."
            self._clarification_chat_history.append({"role": "assistant", "content": response})
            return response
        finally:
            self._num_clarification_turns += 1

    def exec_code(self, complete_code: str) -> str:
        return EvalPlusDockerInstanceEvaluator(self._problem, None).exec_code(complete_code)


class ClarificationEnvironment:
    def __init__(self, *args, **kwargs):
        protected_env = _ClarificationEnvironment(*args, **kwargs)
        object.__setattr__(self, "_env", protected_env)

    def __getattribute__(self, name):
        if name.startswith("_"):
            raise RuntimeError(f"Access to '{name}' is not permitted")
        env = object.__getattribute__(self, "_env")
        return getattr(env, name)

    def __setattr__(self, name, value):
        raise RuntimeError("Environment is read-only for participants")
