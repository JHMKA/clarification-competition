"""
Interactive CLI for The Clarification Challenge.

Chat with a coding assistant that either asks a clarification question or
returns an implementation attempt.

    python chat.py [CLARIFIER]                      # start an empty session
    python chat.py [CLARIFIER] --dataset_path data/mbpp_demo_test.jsonl --task_id 42

Plain text is sent to the assistant, anything starting with "/" is a command
(type /help to see them).

Structure
---------
  Response types        what the assistant can answer with
  Assistant             the interface you implement (see PlaceholderAssistant)
  Command registry      add a new command with @command(...)
  ChatCLI               input loop, rendering, session state
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import os
import queue
import re
import sys
import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Union

import fire
from rich._spinners import SPINNERS
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from clarify.baselines.base import ClarificationAlgorithmBase
from clarify.env import (
    ClarificationConfiguration,
    TooManyQuestionException,
    _ClarificationEnvironment,
)
from clarify.llm import LanguageModel

console = Console()

# A custom "jumping dots" spinner used while the assistant is thinking.
SPINNERS["jumping_dots"] = {
    "interval": 110,
    "frames": ["●∘∘∘", "∘●∘∘", "∘∘●∘", "∘∘∘●", "∘∘●∘", "∘●∘∘"],
}

# ---------------------------------------------------------------------------
# Algorithm loading
# ---------------------------------------------------------------------------


def _module_name_for_path(path_to_algorithm: str) -> str:
    # stable, collision-resistant name — avoids clobbering real modules
    # or colliding with another algorithm file that happens to share a basename
    digest = hashlib.sha1(os.path.abspath(path_to_algorithm).encode()).hexdigest()[:12]
    return f"_clarification_algo_{digest}"


def _load_clarification_algorithm(path_to_algorithm: str):
    """
    Parses and compiles the Python file at path_to_algorithm, executing it
    into its own module namespace, then returns the ClarificationAlgorithmBase
    subclass defined in that file.
    """
    if not os.path.exists(path_to_algorithm):
        raise FileNotFoundError(f"{path_to_algorithm} does not exist.")

    module_name = _module_name_for_path(path_to_algorithm)

    # already loaded in this process (e.g. earlier task in the same worker)
    if module_name in sys.modules:
        module = sys.modules[module_name]
    else:
        spec = importlib.util.spec_from_file_location(module_name, path_to_algorithm)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module  # register BEFORE exec, for pickle/dataclasses/etc.
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[module_name]  # don't leave a half-initialized module registered
            raise

    candidates = [
        obj
        for _, obj in inspect.getmembers(module, inspect.isclass)
        if issubclass(obj, ClarificationAlgorithmBase)
        and obj is not ClarificationAlgorithmBase
        and obj.__module__ == module_name  # exclude re-imported subclasses from elsewhere
    ]

    if not candidates:
        raise ValueError(f"No ClarificationAlgorithmBase subclass defined in {path_to_algorithm}.")
    if len(candidates) > 1:
        raise ValueError(
            f"Expected exactly one ClarificationAlgorithmBase subclass in "
            f"{path_to_algorithm}, found: {[c.__name__ for c in candidates]}"
        )

    return candidates[0]


# ---------------------------------------------------------------------------
# Response types
# ---------------------------------------------------------------------------


@dataclass
class ClarificationQuestion:
    """The assistant needs more information before it can implement."""

    question: str


@dataclass
class Response:
    """The assistant gave a regular non-coding answer."""

    text: str


@dataclass
class Implementation:
    """The assistant is confident enough to propose code."""

    code: str
    language: str = "python"
    explanation: str = ""


AssistantResponse = Union[ClarificationQuestion, Implementation, Response]

Message = dict[str, str]  # {"role": "user" | "assistant", "content": str}


# ---------------------------------------------------------------------------
# Assistant interface
# ---------------------------------------------------------------------------


class Assistant:
    """
    Base class for the assistant. Subclass and implement `respond`.

    `history` is the full conversation so far (oldest first). The first user
    message is the task prompt, later user messages are answers to
    clarification questions or additional instructions.
    """

    name: str = "assistant"

    def respond(self, history: list[Message]) -> AssistantResponse:
        raise NotImplementedError


def response_to_text(response: AssistantResponse) -> str:
    """Serialize a response into the plain text stored in the chat history."""
    if isinstance(response, Response):
        return response.text
    if isinstance(response, ClarificationQuestion):
        return response.question
    parts = [f"```{response.language}\n{response.code}\n```"]
    if response.explanation:
        parts.append(response.explanation)
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Command registry
# ---------------------------------------------------------------------------

CommandHandler = Callable[["ChatCLI", list[str]], None]


@dataclass
class Command:
    name: str
    help: str
    usage: str
    handler: CommandHandler
    aliases: tuple[str, ...] = ()


COMMANDS: dict[str, Command] = {}


def command(name: str, help: str, usage: str = "", aliases: tuple[str, ...] = ()):
    """Register a method of ChatCLI as a slash command."""

    def decorator(fn: CommandHandler) -> CommandHandler:
        cmd = Command(name, help, usage or f"/{name}", fn, aliases)
        for key in (name, *aliases):
            COMMANDS[key] = cmd
        return fn

    return decorator


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


@dataclass
class Session:
    task_id: str | None = None
    history: list[Message] = field(default_factory=list)
    responses: list[AssistantResponse] = field(default_factory=list)

    @property
    def num_questions(self) -> int:
        return sum(isinstance(r, ClarificationQuestion) for r in self.responses)

    @property
    def num_implementations(self) -> int:
        return sum(isinstance(r, Implementation) for r in self.responses)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

IMPLEMENT = "--<IMPLEMENT>---"


class ChatCLI:
    PROMPT = "[bold cyan]you ›[/] "

    def __init__(self, assistant: Assistant, dataset_path: str | None = None):
        self.assistant = assistant
        self.session = Session()
        self.dataset: dict[str, dict] = {}
        self.running = True
        if dataset_path:
            self.load_dataset(dataset_path)

    # -- main loop ---------------------------------------------------------

    def run(self) -> None:
        self.print_welcome()
        while self.running:
            try:
                user_input = console.input(self.PROMPT).strip()
            except (EOFError, KeyboardInterrupt):
                console.print()
                break

            if not user_input:
                continue
            if user_input.startswith("/"):
                self.dispatch_command(user_input)
            else:
                self.send_message(user_input)

        console.print("[dim]Bye! 👋[/]")

    def dispatch_command(self, raw: str) -> None:
        name, *args = raw[1:].split()
        cmd = COMMANDS.get(name.lower())
        if cmd is None:
            self.warn(f"Unknown command [bold]/{name}[/]. Type /help for a list.")
            return
        cmd.handler(self, args)

    # -- chatting ----------------------------------------------------------

    def send_message(self, content: str) -> None:
        self.session.history.append({"role": "user", "content": content})

        with console.status(
            f"[magenta]{self.assistant.name} is thinking[/]", spinner="jumping_dots"
        ):
            try:
                response = self.assistant.respond(self.session.history)
            except Exception as e:  # keep the session alive on assistant errors
                self.session.history.pop()
                traceback.print_exc()
                self.error(f"Assistant failed: {e!r}")
                return

        self.session.responses.append(response)
        self.session.history.append({"role": "assistant", "content": response_to_text(response)})
        self.render_response(response)

    # -- rendering ---------------------------------------------------------

    def render_response(self, response: AssistantResponse) -> None:
        if isinstance(response, ClarificationQuestion):
            console.print(
                Panel(
                    Text(response.question),
                    title="❓ Clarification question",
                    subtitle="[dim]answer below, or /implement to force code[/]",
                    border_style="green",
                )
            )
        elif isinstance(response, Response):
            console.print(
                Panel(
                    Text(response.text),
                    title="Assistant's Answer",
                    border_style="green",
                )
            )
        else:
            body = Syntax(
                response.code, response.language, line_numbers=True, theme="monokai", word_wrap=True
            )
            console.print(Panel(body, title="🛠  Implementation attempt", border_style="purple"))
            if response.explanation:
                console.print(
                    Panel(Markdown(response.explanation), title="Explanation", border_style="dim")
                )
        console.print()

    def render_history(self) -> None:
        if not self.session.history:
            self.info("No messages yet.")
            return
        for msg in self.session.history:
            if msg["role"] == "user":
                console.print(
                    Panel(
                        Text(msg["content"]), title="you", border_style="cyan", title_align="left"
                    )
                )
            else:
                console.print(
                    Panel(
                        Markdown(msg["content"]),
                        title=self.assistant.name,
                        border_style="magenta",
                        title_align="left",
                    )
                )

    def print_welcome(self) -> None:
        console.print(
            Panel.fit(
                "[bold]The Clarification Challenge[/] — interactive playground\n\n"
                "Describe what you want implemented. The assistant either asks a\n"
                "[green]clarification question[/] or returns an [purple]implementation[/].\n"
                "Type [bold]/help[/] for commands, [bold]/quit[/] to leave.",
                border_style="bright_blue",
            )
        )
        self.info(f"Assistant: [bold]{self.assistant.name}[/]")
        if self.dataset:
            self.info(f"{len(self.dataset)} tasks loaded — pick one with /task <id>")
        console.print()

    # -- dataset -----------------------------------------------------------

    def load_dataset(self, path: str) -> None:
        with open(path) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        self.dataset = {str(row["task_id"]): row for row in rows}

    def start_task(self, task_id: str) -> None:
        task = self.dataset.get(task_id)
        if task is None:
            self.warn(f"Task [bold]{task_id}[/] not found. Use /tasks to list them.")
            return
        self.session = Session(task_id=task_id)
        console.print(Rule(f"Task {task_id}", style="bright_blue"))
        self.send_message(task["prompt"])

    # -- output helpers ----------------------------------------------------

    @staticmethod
    def info(text: str) -> None:
        console.print(f"[dim]ℹ[/] {text}")

    @staticmethod
    def warn(text: str) -> None:
        console.print(f"[yellow]⚠[/] {text}")

    @staticmethod
    def error(text: str) -> None:
        console.print(f"[bold red]✖[/] {text}")

    # -- commands ----------------------------------------------------------

    @command("help", "Show this help", aliases=("h", "?"))
    def cmd_help(self, args: list[str]) -> None:
        table = Table(box=None, show_header=False, padding=(0, 2))
        seen = set()
        for cmd in COMMANDS.values():
            if cmd.name in seen:
                continue
            seen.add(cmd.name)
            aliases = f" [dim]({', '.join('/' + a for a in cmd.aliases)})[/]" if cmd.aliases else ""
            table.add_row(f"[bold]{cmd.usage}[/]{aliases}", cmd.help)
        console.print(Panel(table, title="Commands", border_style="bright_blue"))

    @command("new", "Start a fresh conversation", aliases=("reset", "clear"))
    def cmd_new(self, args: list[str]) -> None:
        self.session = Session()
        console.clear()
        self.print_welcome()

    @command(
        "implement", "Ask the assistant to stop asking and produce code now", aliases=("impl", "go")
    )
    def cmd_implement(self, args: list[str]) -> None:
        if not self.session.history:
            self.warn("Nothing to implement yet — describe a task first.")
            return
        self.send_message(IMPLEMENT)

    @command("trace", "Show the last trace of clarification", aliases=("t", "tr"))
    def cmd_trace(self, args: list[str]) -> None:
        try:
            trace = self.assistant.env.trace()
        except Exception:
            self.warn("No trace available — interact with the assistant first.")
            return

        for i, conversation in enumerate(trace):
            console.print(Text(f"==== Conversation #{i} ===="))
            for msg in conversation:
                if msg["role"] == "user":
                    console.print(
                        Panel(
                            Text(msg["content"]),
                            title="you",
                            border_style="cyan",
                            title_align="left",
                        )
                    )
                else:
                    console.print(
                        Panel(
                            Markdown(msg["content"]),
                            title=self.assistant.name,
                            border_style="magenta",
                            title_align="left",
                        )
                    )

    @command("task", "Load a task from the dataset and send its prompt", usage="/task <id>")
    def cmd_task(self, args: list[str]) -> None:
        if not self.dataset:
            self.warn("No dataset loaded. Start with --dataset_path <file.jsonl>.")
            return
        if not args:
            self.warn("Usage: /task <id>")
            return
        self.start_task(args[0])

    @command("tasks", "List available task ids", usage="/tasks [n]")
    def cmd_tasks(self, args: list[str]) -> None:
        if not self.dataset:
            self.warn("No dataset loaded.")
            return
        limit = int(args[0]) if args else 20
        table = Table(
            title=f"Tasks ({len(self.dataset)} total, showing {min(limit, len(self.dataset))})"
        )
        table.add_column("id", style="bold")
        table.add_column("prompt")
        for task_id, row in list(self.dataset.items())[:limit]:
            table.add_row(task_id, Text(row["prompt"].strip().replace("\n", " ")[:100]))
        console.print(table)

    @command("history", "Show the full conversation", aliases=("hist",))
    def cmd_history(self, args: list[str]) -> None:
        self.render_history()

    @command("stats", "Show session statistics")
    def cmd_stats(self, args: list[str]) -> None:
        s = self.session
        table = Table(box=None, show_header=False)
        table.add_row("task", s.task_id or "[dim]free chat[/]")
        table.add_row("user messages", str(sum(m["role"] == "user" for m in s.history)))
        table.add_row("clarification questions", str(s.num_questions))
        table.add_row("implementation attempts", str(s.num_implementations))
        console.print(Panel(table, title="Session", border_style="bright_blue"))

    @command("save", "Save the conversation as JSON", usage="/save <file>")
    def cmd_save(self, args: list[str]) -> None:
        if not args:
            self.warn("Usage: /save <file>")
            return
        path = Path(args[0])
        payload = {
            "task_id": self.session.task_id,
            "assistant": self.assistant.name,
            "history": self.session.history,
        }
        path.write_text(json.dumps(payload, indent=2))
        self.info(f"Saved to [bold]{path}[/]")

    @command("quit", "Exit", aliases=("exit", "q"))
    def cmd_quit(self, args: list[str]) -> None:
        self.running = False


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

ROUTER_SYSTEM_PROMPT = """\
You are the front desk of a coding assistant. You receive user messages and \
decide what the user intends.

A message is a CODING TASK if the user asks for a program, function, class, \
script, or algorithm to be written or implemented — even if the description is \
vague or incomplete. Questions about code, explanations, opinions, greetings, \
or anything else are NOT a coding task.

Carefully read the complete chat history but focus on the last user message. 
If the user references or relates to something that was said before, incorporate
this in your response. 

Respond with a single JSON object and nothing else — no prose, no Markdown fences.

If it is a coding task:
{"kind": "task",
 "prompt": "<the task description, verbatim or lightly cleaned; keep every \
requirement, example, and constraint the user gave; do NOT add requirements or \
resolve ambiguities>",
 "entry_point": "<name of the function/class the user must implement>"}

Rules for "entry_point":
- Use the name if the user states it (e.g. "write is_prime(n)", "def solve(...)", \
"a function called merge").
- If the user describes the task but never names the function, try to infer the function name from context.

If it is NOT a coding task:
{"kind": "chat",
 "answer": "<a helpful, concise reply to the user, in the user's language>"}
"""


UNKNOWN_ENTRY_POINT = "unknown"

_JSON_OBJECT = re.compile(r"\{.*\}", re.S)
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def parse_router_output(raw: str, original_message: str) -> tuple[str, str | None]:
    match = _JSON_OBJECT.search(raw.replace("```json", "").replace("```", ""))
    if match is None:
        return raw.strip(), None

    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return raw.strip()

    kind = str(data.get("kind", "")).lower()

    if kind == "task":
        prompt = str(data.get("prompt") or "").strip() or original_message.strip()
        entry_point = str(data.get("entry_point") or "").strip()
        if not _IDENT.match(entry_point):  # "", "foo(x)", "unknown"...
            entry_point = UNKNOWN_ENTRY_POINT
        return prompt, entry_point

    if kind == "chat":
        return str(data.get("answer") or "").strip() or raw.strip(), None

    return raw.strip(), None


# ---------------------------------------------------------------------------
# Chat Assistant
# ---------------------------------------------------------------------------


@dataclass
class Question:
    text: str


@dataclass
class Finished:
    result: str


@dataclass
class Failed:
    error: BaseException


class InteractiveEnvironment(_ClarificationEnvironment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.to_cli: queue.Queue = queue.Queue()
        self.from_cli: queue.Queue = queue.Queue()
        self.implement_requested = threading.Event()
        self._trace = []

    def llm(self, messages: list[dict[str, str]] | str) -> str:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        response = super().llm(messages)
        self._trace.append(list(messages) + [{"role": "assistant", "content": response}])
        return response

    def trace(self) -> list[list[dict[str, Any]]]:
        return self._trace

    def can_ask(self):
        return not self.implement_requested.is_set()

    def ask_human(self, question: str) -> str:
        if not self.can_ask():
            raise TooManyQuestionException(
                "You exceeded the clarification budget by asking too many or overly complex questions."
            )
        self.to_cli.put(Question(question))
        answer = self.from_cli.get()
        if IMPLEMENT in answer:
            self.implement_requested.set()
            return "Please do not ask further questions; provide your best implementation now."
        return answer


class AlgorithmRunner:
    def __init__(self, algorithm, env: InteractiveEnvironment, problem: dict):
        self.env = env
        self.thread = threading.Thread(target=self._target, args=(algorithm, problem), daemon=True)

    def _target(self, algorithm, problem):
        try:
            self.env.to_cli.put(Finished(algorithm.run(self.env, problem)))
        except BaseException as e:
            self.env.to_cli.put(Failed(e))

    def start(self):
        self.thread.start()

    def next_event(self, timeout=0.1):
        """Non-blocking poll so the caller can keep the spinner alive."""
        try:
            return self.env.to_cli.get(timeout=timeout)
        except queue.Empty:
            return None

    def answer(self, text: str):
        self.env.from_cli.put(text)


class ChatAssistant(Assistant):
    name: str = "assistant"

    def __init__(
        self,
        config: ClarificationConfiguration,
        clarification_algorithm: ClarificationAlgorithmBase,
    ):
        self.config = config
        self.clarification_algorithm = clarification_algorithm
        self.name = clarification_algorithm.__class__.__name__

        self._llm = LanguageModel(self.config.language_model, temperature=self.config.temperature)

        self._coding_session: AlgorithmRunner | None = None
        self.env: InteractiveEnvironment = None

    def _parse_init_message(
        self, message: str, history: list[dict[str, str]]
    ) -> tuple[str, str | None]:
        raw = self._llm(
            [
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            ]
            + history
        )
        return parse_router_output(raw, original_message=message)

    def _init_session(self, problem: dict[str, str]):
        self.env = InteractiveEnvironment(self.config, problem)
        self._coding_session = AlgorithmRunner(self.clarification_algorithm, self.env, problem)
        self._coding_session.start()

    def respond(self, history: list[Message]) -> AssistantResponse:
        message = history[-1]["content"]

        if self._coding_session is None:
            prompt_or_respond, entry_point = self._parse_init_message(message, history=history)
            if not entry_point:
                return Response(prompt_or_respond)

            self._init_session({"prompt": prompt_or_respond, "entry_point": entry_point})
        else:
            # This must be an answer to a coding event
            self._coding_session.answer(message)

        event = self._coding_session.next_event()
        while event is None:
            event = self._coding_session.next_event()

        if isinstance(event, Question):
            return ClarificationQuestion(event.text)

        if isinstance(event, Finished):
            self._coding_session = None
            return Implementation(event.result)
        else:
            self._coding_session = None
            raise event.error


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(
    clarify_py: str,
    dataset_path: str | None = None,
    task_id: str | None = None,
    model_name: str = "openai/gpt-4.1-mini",
    temperature: float = 0.7,
    decision_model: str | None = None,
    **kwargs: Any,
) -> None:

    config = ClarificationConfiguration(
        language_model=model_name,
        temperature=temperature,
        decision_model=decision_model,
        max_clarification_turns=1000,
        max_clarification_budget=2.0,
        max_prompt_budget=2.0,
    )

    # Load clarification algorithm and use kwargs as config options
    clarification_algorithm_class = _load_clarification_algorithm(clarify_py)

    assistant = ChatAssistant(config, clarification_algorithm_class(kwargs))
    cli = ChatCLI(assistant, dataset_path=dataset_path)
    if task_id is not None:
        cli.print_welcome()
        cli.start_task(str(task_id))
    cli.run()


if __name__ == "__main__":
    fire.Fire(main)
