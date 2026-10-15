# SPDX-FileCopyrightText: 2026 Pengchao Zheng <pczheng@mail.ustc.edu.cn>
#
# SPDX-License-Identifier: MIT

"""ClariFact V1.1: clarify one consequential fact before solving.

The model decides whether one missing behavioral fact merits a question. A solver
then receives the task and the actual answer without seeing a speculative first
implementation. A final review may correct a concrete error in that solution.

Team: KeepGoing-Z
Team Members: Pengchao Zheng, Mengyi Hu
Main Contact: pczheng@mail.ustc.edu.cn
"""

from __future__ import annotations

import ast
import keyword
import re

from clarify.baselines import ClarificationAlgorithmBase
from clarify.env import (
    ClarificationEnvironment,
    LimitsExceededException,
    TooManyQuestionException,
)

_OUTPUT_FORMAT_REMINDER = (
    "Important: Follow the *Output Format* in your system instructions exactly. "
    "Write ANSWERS=`<answer_to_question>`, with one pair of outer backticks "
    "around the answer. Avoid backticks inside the answer text so the SDK "
    "parser does not end the answer at the wrong position. Preserve the "
    "factual content of your answer."
)

_GATE_SYSTEM = """You are about to implement a Python function for a user.
Before writing code, decide whether one unknown behavioral fact could change a
normal output. Privately consider two plausible behaviors on the same input:
if the task and its examples settle the choice, there is nothing to ask.
Include the public call and return contract in this judgment. If the prose,
shown signature, and examples leave the number or meaning of arguments, the
accepted input shape, or the returned value or shape genuinely unresolved,
consider which single fact would distinguish two plausible normal calls or
outputs. A different placeholder function name is already resolved by the
separately supplied entry point; do not ask about that alone. Do not request
a whole signature or bundle several independent contract details.
If the task and examples do not settle the choice, request the single fact
whose answer would most improve the implementation. It must be answerable
precisely without asking for tests,
the implementation, or a broad restatement of requirements.
Reply with one line only: either IMPLEMENT or ASK: followed by one question.
""".strip()

_SOLVE_SYSTEM = """Implement the requested Python function as a complete,
self-contained module. The required entry point is supplied separately from the
task text: expose that exact name even if the text uses a different placeholder
function name. Preserve the requested arguments, examples, and explicit
constraints. Use a clarification answer only for the precise fact
it resolves; an explicit factual answer can resolve a conflict in the task.
Do not invent a requirement from a vague answer or assume familiar benchmark
behavior overrides the user's text. Include necessary imports. Return only one
complete fenced Python code block, without tests or commentary.
""".strip()

_REVIEW_SYSTEM = """Check a Python implementation against the user's task and
any actual clarification. Look for a concrete behavioral error, missing import,
or violation of an explicit example. Do not invent hidden tests or new
requirements. If the implementation already follows the available specification,
reply KEEP. Otherwise return one complete corrected fenced Python code block.
Do not make stylistic changes and do not include an explanation.
""".strip()


def _question(response: str | None, max_chars: int) -> str | None:
    if not isinstance(response, str):
        return None
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    if len(lines) != 1:
        return None
    line = lines[0]
    if line.upper() == "IMPLEMENT":
        return None
    if line.upper().startswith("ASK:"):
        line = line[4:].strip()
    if not line or len(line) > max_chars or line.count("?") + line.count("？") != 1:
        return None
    return line if line.endswith(("?", "？")) else None


def _code(response: str | None, entry_point: str) -> str | None:
    if not isinstance(response, str):
        return None
    blocks = re.findall(
        r"```(?:python3|python|py)?[ \t]*\r?\n([\s\S]*?)```", response, re.IGNORECASE
    )
    for candidate in (*blocks, response):
        source = candidate.strip()
        if not source or "```" in source:
            continue
        try:
            tree = ast.parse(source)
            compile(tree, "<candidate>", "exec")
        except (SyntaxError, ValueError, TypeError, RecursionError):
            continue
        if any(
            isinstance(node, ast.FunctionDef) and node.name == entry_point for node in tree.body
        ):
            return source
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        if (
            len(functions) == 1
            and entry_point.isidentifier()
            and not keyword.iskeyword(entry_point)
            and not _bound_elsewhere(tree, entry_point)
        ):
            # A task may show a generic function name while the SDK supplies
            # the public entry point. Keep the implementation and its recursion.
            aliased = f"{source}\n\n{entry_point} = {functions[0].name}"
            try:
                compile(aliased, "<candidate-with-alias>", "exec")
            except (SyntaxError, ValueError, TypeError, RecursionError):
                continue
            return aliased
    return None


def _bound_elsewhere(tree: ast.Module, name: str) -> bool:
    """Avoid replacing an existing binding when a one-function alias is uncertain."""
    for statement in tree.body:
        if isinstance(statement, ast.FunctionDef):
            continue
        for node in ast.walk(statement):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name == name:
                    return True
            elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                if node.id == name:
                    return True
            elif isinstance(node, ast.alias):
                if (node.asname or node.name.split(".", 1)[0]) == name:
                    return True
    return False


def _fenced(source: str) -> str:
    return f"```python\n{source}\n```"


class ClariFact(ClarificationAlgorithmBase):
    """A model-led, single-question solver with one optional correctness review."""

    DEFAULT_CONFIG = {
        "max_llm_calls": 4,  # Gate, fresh solution, optional format repair, and review.
        "max_question_chars": 320,  # Keep a question answerable as one behavioral fact.
    }

    def run(self, env: ClarificationEnvironment, problem: dict[str, str]) -> str:
        for name in self.DEFAULT_CONFIG:
            value = self.config[name]
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")

        prompt = problem["prompt"]
        entry_point = problem["entry_point"]
        calls = 0

        def ask_model(system: str, user: str, optional: bool = False) -> str | None:
            nonlocal calls
            if calls >= self.config["max_llm_calls"] or env.prompt_cost >= env.prompt_budget:
                return None
            calls += 1
            try:
                return env.llm(
                    [{"role": "system", "content": system}, {"role": "user", "content": user}]
                )
            except LimitsExceededException:
                return None
            except Exception:
                if optional:
                    return None
                raise

        question = None
        answer = None
        if env.can_ask():
            gate = ask_model(
                _GATE_SYSTEM,
                f"Function to implement: {entry_point}\n\nTask:\n{prompt}",
                optional=True,
            )
            question = _question(gate, self.config["max_question_chars"])
            if question and env.can_ask():
                try:
                    reply = env.ask_human(f"{question}\n\n{_OUTPUT_FORMAT_REMINDER}")
                except TooManyQuestionException:
                    reply = None
                if isinstance(reply, str) and reply.strip() not in {
                    "",
                    "Irrelevant question.",
                    "Cannot answer the given question.",
                }:
                    answer = reply.strip()

        task = f"Task:\n{prompt}"
        if answer is not None:
            task += f"\n\nClarification question:\n{question}\n\nUser's answer:\n{answer}"
        task += (
            f"\n\nRequired public entry point: {entry_point}. "
            "The returned module must expose this exact top-level callable name."
        )

        response = ask_model(_SOLVE_SYSTEM, task)
        code = _code(response, entry_point)
        if code is None and response is not None:
            repair = ask_model(
                _SOLVE_SYSTEM,
                task
                + "\n\nYour previous response was not a complete, compilable Python module "
                + f"defining {entry_point!r}. Return a corrected complete Python code block."
                + "\n\nPrevious response:\n"
                + response,
                optional=True,
            )
            code = _code(repair, entry_point)
        if code is None:
            raise ValueError("Model did not produce a complete implementation of the entry point")

        review = ask_model(
            _REVIEW_SYSTEM,
            task + "\n\nCandidate implementation:\n" + _fenced(code),
            optional=True,
        )
        return _fenced(_code(review, entry_point) or code)
