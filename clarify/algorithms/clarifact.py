# SPDX-FileCopyrightText: 2026 Pengchao Zheng <pczheng@mail.ustc.edu.cn>
#
# SPDX-License-Identifier: MIT

"""ClariFact V111: one residual repair after measured incumbent progress.

Preserve the complete inherited clarification and solution path. Observe only
actual parsed executions on unchanged visible checks. Continue once only if the
final inherited incumbent strictly improved the first verified candidate without
losing any prior pass, yet a public failure remains and an inherited execution
slot is available. Pair that exact incumbent with its own report. A replacement
must further improve the same checks without regression; otherwise keep it.
No new tests or expected values, independent candidates, plans, or raw answers
are introduced. Public progress does not establish hidden correctness.
At most 9 algorithm LLM calls (8 inherited plus residual1), at most 3
SDK executions and one permitted SDK clarification. Earlier call/exec subcaps
and SDK dollar budgets remain binding; the new stage can be inactive.

Team: KeepGoing-Z
Team Members: Pengchao Zheng, Mengyi Hu
Main Contact: pczheng@mail.ustc.edu.cn
"""

from __future__ import annotations

import ast
import doctest
import json
import keyword
import math
import re

from clarify.baselines import ClarificationAlgorithmBase
from clarify.env import ClarificationEnvironment, LimitsExceededException, TooManyQuestionException

_OUTPUT_FORMAT_REMINDER = (
    "Important: Follow the *Output Format* in your system instructions exactly. "
    "Write ANSWERS=`<answer_to_question>`, with one pair of outer backticks "
    "around the answer. Avoid backticks inside the answer text so the SDK "
    "parser does not end the answer at the wrong position. Preserve the "
    "factual content of your answer."
)


_SOLVE_SYSTEM = """Implement one compact complete Python module from the entire
original programming task, public entry, and actual usable SDK answer if any.
Return exactly one fenced Python code block. Keep explicit public examples,
formulas, order, input/output and complexity requirements. An absent, empty,
refused, vague, or irrelevant answer creates no new fact and does not confirm
the question premise. A real answer can resolve only the scope it addresses;
retain unaffected commitments. If two explicit commitments conflict and the
actual answer resolves that conflict, use it only for that conflict. Do not
guess a missing constant from a yes/no or vague answer. Do not receive or rely
on the question generator's hypothetical outcomes. Do not confuse the public
entry name with a familiar task's behavior. Reason about examples but do not
claim to have run them. Expose the exact top-level callable, use only existing
packages, and return no test, plan, or commentary.""".strip()

_REVISE_SYSTEM = """Synthesize a complete Python solution from the entire original
specification and the actual clarification, independently of any earlier code.
First reason internally about what behavior the answer actually establishes,
what conflicting original claims it overrides within its scope, and what facts
remain unknown. Then choose an algorithm matching the clarified computation,
public interface and explicit complexity requirements. Preserve unaffected
requirements. An ambiguous or irrelevant answer does not invent missing facts.
Do not merely patch a guessed initial design when the clarified task calls for
a different computation. Never infer semantics only from a familiar entry name.
Return exactly one fenced complete Python module, no tests or commentary.
""".strip()


def _has_simple_public_alias(tree: ast.Module, entry_point: str) -> bool:
    """Accept a plain alias with one visible definition and no opaque top-level effects.

    This is structural admission, not a check of the function's behavior. Complex
    exports remain unsupported rather than executing the module on the host.
    """
    bindings: dict[str, list[int]] = {}
    definitions: dict[str, tuple[int, ast.FunctionDef]] = {}
    aliases: list[tuple[int, str]] = []
    for index, statement in enumerate(tree.body):
        names: list[str] = []
        if isinstance(statement, ast.FunctionDef):
            if statement.decorator_list:
                return False
            names = [statement.name]
            definitions[statement.name] = (index, statement)
        elif isinstance(statement, (ast.Import, ast.ImportFrom)):
            if any(item.name == "*" for item in statement.names):
                return False
            names = [item.asname or item.name.split(".", 1)[0] for item in statement.names]
        elif isinstance(statement, ast.Assign):
            if not all(isinstance(target, ast.Name) for target in statement.targets):
                return False
            names = [target.id for target in statement.targets]
            if not isinstance(statement.value, ast.Name):
                try:
                    ast.literal_eval(statement.value)
                except (ValueError, TypeError, SyntaxError, RecursionError):
                    return False
            if (
                len(statement.targets) == 1
                and names == [entry_point]
                and isinstance(statement.value, ast.Name)
            ):
                aliases.append((index, statement.value.id))
        elif (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        ):
            continue
        else:
            return False
        for name in names:
            bindings.setdefault(name, []).append(index)
    if len(aliases) != 1 or len(bindings.get(entry_point, [])) != 1:
        return False
    index, target = aliases[0]
    definition = definitions.get(target)
    return bool(definition and definition[0] < index and len(bindings.get(target, [])) == 1)


def _has_public_lambda(tree: ast.Module, entry_point: str) -> bool:
    """Admit one clear binding; structural coverage is not functional validation."""
    bindings: dict[str, int] = {}
    found = False
    for statement in tree.body:
        names: list[str] = []
        if isinstance(statement, ast.FunctionDef):
            if statement.decorator_list:
                return False
            names = [statement.name]
        elif isinstance(statement, (ast.Import, ast.ImportFrom)):
            if any(item.name == "*" for item in statement.names):
                return False
            names = [item.asname or item.name.split(".", 1)[0] for item in statement.names]
        elif isinstance(statement, ast.Assign):
            if not all(isinstance(target, ast.Name) for target in statement.targets):
                return False
            names = [target.id for target in statement.targets]
            if entry_point in names and isinstance(statement.value, ast.Lambda):
                if len(names) != 1:
                    return False
                defaults = statement.value.args.defaults + [
                    value for value in statement.value.args.kw_defaults if value is not None
                ]
                try:
                    for value in defaults:
                        ast.literal_eval(value)
                except (ValueError, TypeError, SyntaxError, RecursionError):
                    return False
                found = True
            else:
                try:
                    ast.literal_eval(statement.value)
                except (ValueError, TypeError, SyntaxError, RecursionError):
                    return False
        elif (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        ):
            continue
        else:
            return False
        for name in names:
            bindings[name] = bindings.get(name, 0) + 1
    return found and bindings.get(entry_point) == 1


def _code(response: str | None, entry_point: str, *, allow_alias: bool = True) -> str | None:
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
        if _has_public_lambda(tree, entry_point):
            return source
        if allow_alias and _has_simple_public_alias(tree, entry_point):
            return source
        if not allow_alias:
            continue
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


_DIAGNOSE_SYSTEM = """Audit only the entire visible programming specification and
public entry. Never reconstruct a benchmark task from its name. For at most
three output-defining contract components, distinguish:
- explicit: a general behavioral rule is actually stated and not contradicted;
- sample_only: a few input/output examples illustrate a relation without stating
  the general computation; finite examples alone do not define it uniquely;
- missing: an essential rule, parameter role, output meaning or predicate is
  absent, truncated or locally ambiguous;
- conflicting: visible commitments disagree for the same intended behavior.
Check the whole statement and examples before alleging a gap. Source corruption
categories are not semantic proof. Cite an exact short source quote for every
component, using the incomplete span when the rule itself is absent. Describe
what is known and what answer is needed, without inventing a deleted constant,
recurrence, predicate or expected output. Distinguish core computation gaps
from secondary local details. A coherent core transformation can require
inseparable facts such as its predicate and position rule or recurrence and
base cases. Do not assume those facts and ask only about an edge case.

Alternatives are OPTIONAL. Supply zero when the general rule is unknown. Only
if two concrete readings are supported by visible text, supply exactly two
interpretations, each shaped as {"behavior":"reading", "evidence":"exact supporting source quote"}. These are nonexhaustive hypotheses, never truth or expected values. Do
not invent two stories merely to fill a schema. Conflicting examples can agree
with neither reading. Return only JSON:
{"candidates":[{"status":"missing","scope":"core",
"evidence":"exact source quote","contract":"what this component establishes or leaves unknown",
"question_target":"the required behavioral information",
"interpretations":[]}]}
An explicit component may be reported for grounded confirmation. Empty
candidates are allowed. Do not include code, draft assumptions or probes.
""".strip()

_SELECT_SYSTEM = """Choose one task-specific English clarification using only
the complete visible task, public entry and independent contract coverage.
The audit is a set of source-grounded proposals, not established truth. Select
one output-defining contract whose clarification has the greatest ordinary
valid-input effect. Do not infer facts from a familiar entry name.
1. Missing core or sample-only computation: request its intended general rule
and inseparable parameter/output meanings. If examples are the only evidence,
ask whether their shown outputs are authoritative for that same computation.
Do not invent a core goal or force a choice between unsupported alternatives.
2. Conflicting commitments: briefly identify the conflict and ask the intended
general behavior, allowing a rule outside stated alternatives. Resolve example
authority only as needed for that conflict; do not assume samples always win.
3. Local gap: ask the missing rule. Include contrasting outcomes on a valid
input only if both are reliably derivable from the visible task. Never invent
expected values, invalid input conditions or a deleted constant as a premise.
4. Adequately explicit task: confirm one concrete observable commitment rather
than allege a nonexistent defect or ask generic additional requirements.
Ask one coherent question, not a bundle of unrelated checks. Coupled missing
facts needed to define one computation can be requested together. Only the
actual user's answer can supply new facts. Ask requirements, not algorithms,
code, hidden tests or missing benchmark text. Return one line
ASK: <one concise question?>, ending in one question mark.
""".strip()

_QUESTION_FORMAT_SYSTEM = """Produce one concise English clarification question
from the full visible task and independent contract coverage. Preserve the
substantive intent of the previous question when supported, but remove its
analysis and formatting. Request an unknown general rule when needed rather
than inventing two alternatives. Preserve a coherent core-contract request
when inseparable facts define the same computation. Otherwise ask one concrete
output-affecting fact, or confirm an explicit task-specific behavior if the
task is complete. Do not request additional requirements in general, tests,
code or hidden information. Return exactly one line: ASK: <question?>""".strip()

_QUESTION_RECOVERY_SYSTEM = """Review ONE already selected clarification before
the sole SDK question is sent. Use the complete visible task and public entry.
Coverage proposals are fallible context, never source facts. Do not infer a
benchmark original, hidden rule, value, predicate, table, example or answer.

Check only whether a truthful literal answer could satisfy the candidate's
wording while leaving THE SAME intended single fact unidentified. For example,
rejecting a guessed value may not supply the intended value; confirming an
undefined label may not supply its definition. This is a logical question about
answer form, not a prediction of the actual user's answer or a question score.
Do not output any simulated answer, expected output, code, test, confidence or
PASS claim. A binary question is adequate when its two source-supported cases
really exhaust the same fact. Other independent unknowns are NOT evidence that
this question fails: a parity answer need not supply ordering or endpoints.

Return KEEP unless you can identify one concrete recovery gap and repair ONLY
that gap while preserving the candidate's substantive target and atomicity.
Do not retarget to another missing rule or widen a representation question into
a mapping/table, add independent conditions, bundle facts, or force new A/B
choices. For an already explicit target, preserve a supported confirmation.
When repair would need another fact, or the target is unclear, return KEEP.
Exact quotations establish provenance only, not entailment or atomicity.

Return only {"decision":"KEEP"}, or exactly:
{"decision":"REWRITE", "question_quote":"the entire exact candidate question",
 "source_quote":"one exact short span of the original prompt",
 "same_fact":"the single fact already sought by the old question",
 "recovery_gap":"why a literal answer may leave that same fact unidentified",
 "question":"ASK: one concise English question seeking that same fact?"}.
The source quote must occur in the original prompt, at most 600 characters.
Keep same_fact at most 400 characters and recovery_gap at most 800 characters.
No new fact is established by this review or by a question premise.
""".strip()


def _question_recovery(response, prompt, question, limit):
    """Admit bounded source membership only; semantic claims remain fallible."""
    if not isinstance(response, str) or len(response) > 4000:
        return None
    try:
        item = json.loads(response.strip())
    except (ValueError, TypeError, RecursionError):
        return None
    if not isinstance(item, dict) or item == {"decision": "KEEP"}:
        return None
    required = {
        "decision",
        "question_quote",
        "source_quote",
        "same_fact",
        "recovery_gap",
        "question",
    }
    if set(item) != required or item.get("decision") != "REWRITE":
        return None
    if item["question_quote"] != question:
        return None
    for name, maximum in (("source_quote", 600), ("same_fact", 400), ("recovery_gap", 800)):
        value = item[name]
        if (
            not isinstance(value, str)
            or not value.strip()
            or len(value) > maximum
            or "```" in value
        ):
            return None
    if item["source_quote"] not in prompt:
        return None
    replacement = _question(item["question"], limit)
    if replacement is None or replacement == question:
        return None
    return replacement


_EVIDENCE_SYSTEM = """Check which literal public examples remain
authoritative after the actual clarification. You are not judging code. Read
the complete original task, question and real answer. An answer can resolve a
contradiction and make old examples obsolete. Keep an original example only
when its expected behavior remains consistent with the actual answer; omit an
example when the answer overrides it or applicability is uncertain. Never
rewrite an example, invent an expected value or infer a hidden requirement.

Separately, normalize at most two direct literal witnesses already written in
the actual answer. Normalize only when every nonempty answer line is a complete
literal call/result proposition, with no surrounding prose or qualification.
Return every such line, up to two; otherwise return no witnesses. Each quote
must copy one complete trimmed answer line:
exactly one direct call of the required public entry with literal arguments,
then ==, ->, =>, returns, returned, result is, output is, should be, or gives,
then exactly one Python literal result and optional terminal punctuation. Do
not include a lead-in, explanation, negation, condition, alternative result or
additional proposition in the quote. Copy the whole quote, call and result
verbatim from the answer. Do not infer an input from the question, complete an
omitted function call, translate prose into a new literal, or generate a test
or expected value.

Return only JSON:
{"keep":[0,2],"answer_witnesses":[
 {"quote":"continuous exact answer quote",
  "call":"entry(literal arguments)","expected":"literal result"}]}
An empty list is legitimate. Indices refer only to the supplied original-task
examples. A malformed witness must not affect the keep decision.""".strip()

_REPAIR_SYSTEM = """Locally repair the supplied Python module only using a real
execution failure and its visible source evidence. Sources are the complete
original task, actual usable SDK answer and explicit literal public examples
that remain applicable after that answer. The answer takes precedence for the
scope it resolves. A probe observation is not an oracle and must not become a
requirement. Preserve unaffected semantics and every previously passing public
obligation. Do not guess hidden rules or change the interface to accept
invented inputs. If the alleged failure rests on a conflict or uncertain
example, return exactly KEEP_CANDIDATE. Otherwise return exactly one complete
fenced Python module exposing the required public entry. Do not add a general
argument-conversion wrapper or tests to the returned implementation.""".strip()


def _json_object(response: str | None) -> dict | None:
    if not isinstance(response, str) or len(response) > 24000:
        return None
    text = response.strip()
    candidates = [text]
    candidates.extend(re.findall(r"```(?:json)?\s*\n([\s\S]*?)```", text, re.I))
    for candidate in candidates:
        try:
            value = json.loads(candidate.strip())
        except (ValueError, TypeError, RecursionError):
            continue
        if isinstance(value, dict):
            return value
    return None


def _literal(text: str):
    """Parse only bounded literal data; this never executes candidate code."""
    if not isinstance(text, str) or len(text) > 4000:
        raise ValueError("Literal is absent or too large")
    tree = ast.parse(text, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 512:
        raise ValueError("Literal contains too many nodes")
    return ast.literal_eval(tree)


def _literal_call(node: ast.AST, entry_point: str) -> tuple[tuple, dict] | None:
    """Accept a direct call of the public entry with literal arguments only."""
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == entry_point
    ):
        return None
    if len(node.args) + len(node.keywords) > 24 or len(ast.dump(node)) > 16000:
        return None
    if any(isinstance(arg, ast.Starred) for arg in node.args):
        return None
    if any(item.arg is None for item in node.keywords):
        return None
    names = [item.arg for item in node.keywords]
    if len(set(names)) != len(names):
        return None
    try:
        if sum(1 for _ in ast.walk(node)) > 512:
            return None
        args = tuple(ast.literal_eval(item) for item in node.args)
        kwargs = {item.arg: ast.literal_eval(item.value) for item in node.keywords}
    except (ValueError, TypeError, SyntaxError, RecursionError, MemoryError):
        return None
    if len(repr((args, kwargs))) > 4000:
        return None
    return args, kwargs


def _diagnoses(response: str | None, prompt: str, max_witnesses: int) -> list[dict]:
    """Admit quoted coverage claims, not their unverified semantic conclusions.

    Exact quote membership is a provenance check. It cannot prove a reading is
    logically supported. Missing general rules need no fabricated alternatives.
    """
    value = _json_object(response)
    if value is None or not isinstance(value.get("candidates"), list):
        return []
    accepted = []
    for item in value["candidates"][:max_witnesses]:
        if not isinstance(item, dict):
            continue
        status, scope = item.get("status"), item.get("scope")
        evidence, contract = item.get("evidence"), item.get("contract")
        target = item.get("question_target")
        readings = item.get("interpretations", [])
        if not (
            isinstance(status, str)
            and status in {"explicit", "sample_only", "missing", "conflicting"}
            and isinstance(scope, str)
            and scope in {"core", "local"}
            and isinstance(evidence, str)
            and 0 < len(evidence) <= 400
            and evidence in prompt
            and isinstance(contract, str)
            and 0 < len(contract) <= 600
            and isinstance(target, str)
            and 0 < len(target) <= 600
            and isinstance(readings, list)
            and len(readings) in {0, 2}
        ):
            continue
        alternatives = []
        valid = True
        for reading in readings:
            if not isinstance(reading, dict):
                valid = False
                break
            behavior, support = reading.get("behavior"), reading.get("evidence")
            if not (
                isinstance(behavior, str)
                and 0 < len(behavior) <= 600
                and isinstance(support, str)
                and 0 < len(support) <= 400
                and support in prompt
            ):
                valid = False
                break
            alternatives.append({"behavior": behavior, "evidence": support})
        if (
            not valid
            or len(alternatives) == 2
            and alternatives[0]["behavior"] == alternatives[1]["behavior"]
        ):
            continue
        accepted.append(
            {
                "status": status,
                "scope": scope,
                "evidence": evidence,
                "contract": contract,
                "question_target": target,
                "interpretations": alternatives,
            }
        )
    return accepted


def _question(response: str | None, max_chars: int) -> str | None:
    if not isinstance(response, str):
        return None
    value = _json_object(response)
    if value is not None:
        response = value.get("question", value.get("ask"))
        if not isinstance(response, str):
            return None
    text = response.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    text = re.sub(r"^(?:ASK|QUESTION)\s*:\s*", "", text, flags=re.I).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()
    if "```" in text:
        return None
    text = " ".join(text.split())
    # A literal question-mark character can itself be part of an input.
    control = re.sub(r"([\x27\x22`])[?？]\1", "", text)
    if (
        not text
        or len(text) > max_chars
        or not text.endswith("?")
        or control.count("?") + control.count("？") != 1
    ):
        return None
    lower = text.lower()
    if any(
        phrase in lower
        for phrase in (
            "any additional",
            "other requirements",
            "anything else",
            "hidden test",
            "reference_prompt",
            "human_system_prompt",
        )
    ):
        return None
    return text


def _emergency_question(prompt: str, entry_point: str, max_chars: int) -> str:
    """Confirm a quoted task fragment instead of inventing an unknown fact."""
    fragments = [
        part.strip()
        for part in re.split(r"[\r\n]+|(?<=[.!])\s+", prompt)
        if part.strip() and not part.lstrip().startswith((">>>", "```"))
    ]
    fragment = next((part for part in fragments if len(part) >= 12), prompt.strip())
    # A question in the source would introduce another question mark. Use an
    # exact smaller source span; the full task still reaches the simulated user.
    fragment = re.split(r"[?？]", fragment, maxsplit=1)[0].strip()
    allowance = max(12, min(160, max_chars - len(entry_point) - 90))
    fragment = fragment[:allowance].strip()
    if fragment:
        return (
            f"For {entry_point}, should the implementation follow the task rule "
            f'"{fragment}" as its intended behavior?'
        )
    return f'For the public entry "{entry_point}", what result should the function compute?'


def _example(source: str, expected: str, entry_point: str, origin: str) -> dict | None:
    if len(source) + len(expected) > 8000:
        return None
    try:
        statement = ast.parse(source.strip()).body
        if len(statement) != 1 or not isinstance(statement[0], ast.Expr):
            return None
        call = _literal_call(statement[0].value, entry_point)
        result = _literal(expected.strip())
    except (SyntaxError, ValueError, TypeError, RecursionError, MemoryError):
        return None
    if call is None:
        return None
    args, kwargs = call
    return {
        "args": args,
        "kwargs": kwargs,
        "expected": result,
        "source": origin,
        "evidence": source.strip() + "\n" + expected.strip(),
    }


def _public_examples(
    prompt: str, answer: str | None, entry_point: str, max_examples: int = 8
) -> list[dict]:
    """Extract explicit literal examples, not model-written expected outputs."""
    found = []
    for origin, text in (("prompt", prompt), ("answer", answer or "")):
        # Docstring delimiters terminate the source documentation, not a
        # doctest's expected literal value. Keep all other text untouched.
        doctext = "\n".join(
            "" if line.strip() in {'"""', "'''"} else line for line in text.splitlines()
        )
        try:
            examples = doctest.DocTestParser().get_examples(doctext)
        except ValueError:
            examples = []
        for item in examples:
            example = _example(item.source, item.want, entry_point, origin)
            if example is not None:
                found.append(example)
        for line in text.splitlines():
            expression = line.strip()
            if len(expression) > 8000:
                continue
            if expression.startswith(">>>"):
                expression = expression[3:].strip()
            try:
                nodes = ast.parse(expression).body
            except (SyntaxError, ValueError, TypeError, RecursionError):
                continue
            if len(nodes) != 1:
                continue
            node = nodes[0]
            comparison = (
                node.test
                if isinstance(node, ast.Assert)
                else (node.value if isinstance(node, ast.Expr) else None)
            )
            if not (
                isinstance(comparison, ast.Compare)
                and len(comparison.ops) == 1
                and isinstance(comparison.ops[0], ast.Eq)
                and len(comparison.comparators) == 1
            ):
                continue
            call = _literal_call(comparison.left, entry_point)
            if call is None:
                continue
            try:
                expected_node = comparison.comparators[0]
                if sum(1 for _ in ast.walk(expected_node)) > 512:
                    continue
                expected = ast.literal_eval(expected_node)
            except (ValueError, TypeError, SyntaxError, RecursionError, MemoryError):
                continue
            if len(repr(expected)) > 4000:
                continue
            found.append(
                {
                    "args": call[0],
                    "kwargs": call[1],
                    "expected": expected,
                    "source": origin,
                    "evidence": expression,
                }
            )
    # A direct answer example supersedes a prompt example for exactly the same
    # literal call. Conflicting examples within one source are excluded; neither
    # expectation is an established oracle in that situation.
    by_call: dict[str, list[dict]] = {}
    for item in found:
        by_call.setdefault(repr((item["args"], item["kwargs"])), []).append(item)
    selected = []
    for items in by_call.values():
        answer_items = [item for item in items if item["source"] == "answer"]
        preferred = answer_items or items
        if len({repr(item["expected"]) for item in preferred}) != 1:
            continue
        selected.append(preferred[0])
    # Preserve answer evidence when a long statement has many examples.
    selected.sort(key=lambda item: item["source"] != "answer")
    return selected[:max_examples]


def _applicable_examples(response: str | None, examples: list[dict]) -> list[dict]:
    """A malformed applicability response removes uncertain prompt examples."""
    value = _json_object(response)
    indices = value.get("keep") if value is not None else None
    prompt_indices = {index for index, item in enumerate(examples) if item["source"] == "prompt"}
    if (
        not isinstance(indices, list)
        or any(type(i) is not int for i in indices)
        or len(indices) != len(set(indices))
        or not set(indices) <= prompt_indices
    ):
        indices = []
    keep = set(indices)
    return [
        item for index, item in enumerate(examples) if item["source"] == "answer" or index in keep
    ]


def _answer_witnesses(
    response: str | None, answer: str, entry_point: str, max_witnesses: int
) -> list[dict]:
    """Admit only a literal call/result relation copied from one answer span."""
    value = _json_object(response)
    records = value.get("answer_witnesses") if value is not None else None
    if not isinstance(records, list):
        return []
    # Only a self-contained answer made entirely of direct relation lines may
    # supply normalized witnesses. Surrounding prose can negate or qualify an
    # otherwise affirmative substring, so partial quotations are insufficient.
    answer_lines = [line.strip() for line in answer.splitlines() if line.strip()]
    if not answer_lines or len(answer_lines) > 32:
        return []
    # Inspect all bounded records before applying the output limit. Otherwise a
    # third valid record can contradict the first two and be silently ignored.
    if len(records) > 32:
        return []
    accepted = []
    relation_pattern = (
        r"(?i:==|=>|->|returns?|returned|result\s+is|output\s+is|"
        r"should\s+be|gives)"
    )
    for item in records:
        if not isinstance(item, dict) or set(item) != {"quote", "call", "expected"}:
            continue
        quote, call_text, expected_text = (
            item.get("quote"),
            item.get("call"),
            item.get("expected"),
        )
        if not all(
            isinstance(part, str) and part.strip() for part in (quote, call_text, expected_text)
        ):
            continue
        quote, call_text, expected_text = (quote.strip(), call_text.strip(), expected_text.strip())
        if (
            len(quote) > 1200
            or len(call_text) > 500
            or len(expected_text) > 500
            or quote not in answer_lines
            or "```" in call_text
            or "```" in expected_text
        ):
            continue
        # The complete quote must be one minimal affirmative proposition. A
        # full-string match rejects negation, numeric substrings, unrelated
        # relations, conditions and alternative results after the literal.
        direct_relation = re.compile(
            rf"{re.escape(call_text)}\s*{relation_pattern}\s*"
            rf"{re.escape(expected_text)}\s*[.!]?",
        )
        if direct_relation.fullmatch(quote) is None:
            continue
        try:
            statements = ast.parse(call_text).body
            if len(statements) != 1 or not isinstance(statements[0], ast.Expr):
                continue
            expected_tree = ast.parse(expected_text, mode="eval")
            if (
                ast.get_source_segment(call_text, statements[0].value) != call_text
                or ast.get_source_segment(expected_text, expected_tree.body) != expected_text
            ):
                continue
            call = _literal_call(statements[0].value, entry_point)
            expected = _literal(expected_text)
        except (SyntaxError, ValueError, TypeError, RecursionError, MemoryError):
            continue
        values = [expected, call] if call is not None else [expected]
        while values:
            value = values.pop()
            if value is Ellipsis:
                break
            if isinstance(value, dict):
                values.extend(value.keys())
                values.extend(value.values())
            elif isinstance(value, (list, tuple, set, frozenset)):
                values.extend(value)
        else:
            if call is not None and len(repr(expected)) <= 4000:
                accepted.append(
                    {
                        "args": call[0],
                        "kwargs": call[1],
                        "expected": expected,
                        "source": "answer",
                        "evidence": quote,
                    }
                )
            continue

    # Do not omit a context-bearing line, an unparsed alternative or a conflict
    # from the answer. Output truncation happens only after this coverage check.
    if set(answer_lines) != {item["evidence"] for item in accepted}:
        return []

    by_call: dict[str, list[dict]] = {}
    for item in accepted:
        by_call.setdefault(repr((item["args"], item["kwargs"])), []).append(item)
    result = []
    for items in by_call.values():
        if len({repr(item["expected"]) for item in items}) == 1:
            result.append(items[0])
    return result[:max_witnesses]


def _merge_examples(examples: list[dict], witnesses: list[dict], limit: int) -> list[dict]:
    """Prefer consistent actual-answer evidence for an identical literal call."""
    by_call: dict[str, list[dict]] = {}
    for item in [*examples, *witnesses]:
        by_call.setdefault(repr((item["args"], item["kwargs"])), []).append(item)
    selected = []
    for items in by_call.values():
        answer_items = [item for item in items if item["source"] == "answer"]
        preferred = answer_items or items
        if len({repr(item["expected"]) for item in preferred}) == 1:
            selected.append(preferred[0])
    selected.sort(key=lambda item: item["source"] != "answer")
    return selected[:limit]


_EXECUTION_HARNESS = r"""
import ast as _v30_ast
import copy as _v30_copy
import json as _v30_json
import math as _v30_math
import signal as _v30_signal

def _v30_alarm(_signum, _frame):
    raise TimeoutError("public observation time limit")

def _v30_repr(value):
    try:
        return repr(value)[:1200]
    except BaseException as error:
        return "<unprintable " + type(error).__name__ + ">"

def _v30_equal(actual, expected):
    if (isinstance(actual, (int, float)) and not isinstance(actual, bool)
            and isinstance(expected, (int, float)) and not isinstance(expected, bool)
            and (isinstance(actual, float) or isinstance(expected, float))):
        return _v30_math.isclose(actual, expected, rel_tol=V30_FLOAT_REL_TOL,
                                abs_tol=V30_FLOAT_ABS_TOL)
    if isinstance(actual, (list, tuple)) and type(actual) is type(expected):
        return len(actual) == len(expected) and all(
            _v30_equal(a, b) for a, b in zip(actual, expected))
    if isinstance(actual, dict) and isinstance(expected, dict):
        return (actual.keys() == expected.keys()
                and all(_v30_equal(actual[key], expected[key]) for key in expected))
    if isinstance(actual, (set, frozenset)) and type(actual) is type(expected):
        remaining = list(actual)
        for value in expected:
            index = next((i for i, item in enumerate(remaining)
                          if _v30_equal(item, value)), None)
            if index is None:
                return False
            remaining.pop(index)
        return not remaining
    return bool(actual == expected)

def _v30_run():
    _v30_signal.signal(_v30_signal.SIGALRM, _v30_alarm)
    namespace = {"__name__": "public_candidate"}
    report = {"status": "ok", "entry_ok": False, "records": [], "error": None}
    try:
        _v30_signal.alarm(3)
        exec(compile(V30_SOURCE, "<candidate>", "exec"), namespace)
        fn = namespace.get(V30_ENTRY)
        report["entry_ok"] = callable(fn)
        if not report["entry_ok"]:
            report["status"] = "load_error"
            report["error"] = "Required public entry is not callable"
            return report
    except BaseException as error:
        report["status"] = "load_error"
        report["error"] = type(error).__name__ + ": " + str(error)[:1200]
        return report
    finally:
        _v30_signal.alarm(0)
    cases = _v30_ast.literal_eval(V30_CASES)
    for index, item in enumerate(cases):
        record = {"index": index, "status": "observed", "actual": None,
                  "expected": None, "error": None}
        if V30_VERIFY:
            record["expected"] = _v30_repr(item["expected"])
        try:
            _v30_signal.alarm(2)
            args = _v30_copy.deepcopy(item["args"])
            kwargs = _v30_copy.deepcopy(item["kwargs"])
            actual = fn(*args, **kwargs)
            record["actual"] = _v30_repr(actual)
            if V30_VERIFY:
                record["status"] = "pass" if _v30_equal(actual, item["expected"]) else "fail"
        except BaseException as error:
            record["status"] = "error"
            record["error"] = type(error).__name__ + ": " + str(error)[:1200]
        finally:
            _v30_signal.alarm(0)
        report["records"].append(record)
    return report

print("__NCV30_RESULT__" + _v30_json.dumps(_v30_run(), ensure_ascii=True))
"""


def _execution_script(
    source: str,
    entry_point: str,
    cases: list[dict],
    *,
    verify: bool,
    float_rel_tol: float = 1e-7,
    float_abs_tol: float = 1e-6,
) -> str:
    literals = [
        {
            "args": item["args"],
            "kwargs": item["kwargs"],
            **({"expected": item["expected"]} if verify else {}),
        }
        for item in cases
    ]
    return (
        f"V30_SOURCE = {source!r}\nV30_ENTRY = {entry_point!r}\n"
        f"V30_CASES = {repr(literals)!r}\nV30_VERIFY = {verify!r}\n"
        f"V30_FLOAT_REL_TOL = {float_rel_tol!r}\nV30_FLOAT_ABS_TOL = {float_abs_tol!r}\n"
        + _EXECUTION_HARNESS
    )


def _execution_result(output: str | None, count: int, *, verify: bool) -> dict | None:
    """Unstructured/partial SDK output is infrastructure uncertainty, not WA."""
    if not isinstance(output, str):
        return None
    marker = "__NCV30_RESULT__"
    lines = output.splitlines()
    matching = [line[len(marker) :] for line in lines if line.startswith(marker)]
    if not matching:
        return None
    value = _json_object(matching[-1].strip())
    if value is None or type(value.get("entry_ok")) is not bool:
        return None
    records = value.get("records")
    if not isinstance(records, list):
        return None
    if value.get("status") == "load_error":
        if (
            value["entry_ok"]
            or records
            or not isinstance(value.get("error"), str)
            or not value["error"]
        ):
            return None
        return value
    if (
        value.get("status") != "ok"
        or not value["entry_ok"]
        or len(records) != count
        or "error" not in value
        or value["error"] is not None
    ):
        return None
    valid = {"pass", "fail", "error"} if verify else {"observed", "error"}
    for number, item in enumerate(records):
        if (
            not isinstance(item, dict)
            or type(item.get("index")) is not int
            or item["index"] != number
            or item.get("status") not in valid
            or not {"index", "status", "actual", "expected", "error"} <= item.keys()
            or any(
                item.get(key) is not None and not isinstance(item.get(key), str)
                for key in ("actual", "expected", "error")
            )
        ):
            return None
        if (
            verify
            and not isinstance(item["expected"], str)
            or not verify
            and item["expected"] is not None
        ):
            return None
        if item["status"] == "error":
            if not item["error"]:
                return None
        elif not isinstance(item["actual"], str) or item["error"] is not None:
            return None
    return value


def _failure_count(result: dict) -> int:
    return int(not result["entry_ok"]) + sum(item["status"] != "pass" for item in result["records"])


def _accept_repair(before: dict, after: dict) -> bool:
    if not after["entry_ok"] or _failure_count(after) >= _failure_count(before):
        return False
    previously_passed = {item["index"] for item in before["records"] if item["status"] == "pass"}
    now_passed = {item["index"] for item in after["records"] if item["status"] == "pass"}
    return previously_passed <= now_passed


_V111_RESIDUAL_SYSTEM = """Repair only the remaining demonstrated public
failure in the supplied current Python module. An earlier repair strictly
improved the same checks; that does not establish general correctness. Preserve
all requirements in the complete original task except the exact scope resolved
by the actual SDK answer. Preserve every already passing public check. Use the
actual remaining failure records, not an imagined hidden requirement. Do not
change the checks, invent expected results, or infer facts from an absent answer.
If no grounded repair is justified, return KEEP_CANDIDATE. Otherwise return one
fenced complete Python module with the public entry and no test harness.""".strip()


class ClariFact(ClarificationAlgorithmBase):
    """Contract elicitation plus strictly answer-grounded witness execution."""

    DEFAULT_CONFIG = {
        "max_llm_calls": 9,  # Inherited8; residual1 separately capped.
        "max_exec_calls": 3,  # Upper cap; final public checks and one repaired check only.
        "max_witnesses": 3,  # At most three source-quoted contract components.
        "max_public_examples": 8,  # Literal task/answer examples, never model guesses.
        "max_answer_witnesses": 2,  # Direct call/result relations copied from the answer.
        "max_question_chars": 420,  # Generated question before the SDK format reminder.
        "public_float_rel_tol": 1e-7,  # Public numerical examples may be rounded.
        "public_float_abs_tol": 1e-6,  # Does not change the official evaluation oracle.
    }

    def run(self, env: ClarificationEnvironment, problem: dict[str, str]) -> str:
        for name in self.DEFAULT_CONFIG:
            if name.startswith("public_float_"):
                value = self.config[name]
                if (
                    type(value) not in (int, float)
                    or not math.isfinite(value)
                    or value < 0
                    or value > 1e-3
                ):
                    raise ValueError(f"{name} must be a finite tolerance between 0 and 0.001")
                continue
            if type(self.config[name]) is not int or self.config[name] < 1:
                raise ValueError(f"{name} must be a positive integer")
        prompt, entry_point = problem["prompt"], problem["entry_point"]
        if not entry_point.isidentifier() or keyword.iskeyword(entry_point):
            raise ValueError("Invalid public entry point")
        call_limit = min(self.config["max_llm_calls"], 8)
        exec_limit = min(self.config["max_exec_calls"], 3)
        witness_limit = min(self.config["max_witnesses"], 3)
        example_limit = min(self.config["max_public_examples"], 8)
        answer_witness_limit = min(self.config["max_answer_witnesses"], 2)
        question_limit = min(self.config["max_question_chars"], 420)
        calls = executions = 0
        _v111_records = []
        _v111_extra_calls = 0
        _v111_total_limit = min(self.config["max_llm_calls"], 9)
        original = (
            f"Complete original task:\n{prompt}\n\nRequired public entry point: {entry_point}"
        )

        def ask_model(system: str, user: str) -> str | None:
            nonlocal calls
            if calls >= call_limit or env.prompt_cost >= env.prompt_budget:
                return None
            calls += 1
            try:
                result = env.llm(
                    [{"role": "system", "content": system}, {"role": "user", "content": user}]
                )
            except LimitsExceededException:
                return None
            return result if isinstance(result, str) else None

        def execute(code: str, cases: list[dict], *, verify: bool) -> dict | None:
            nonlocal executions
            if executions >= exec_limit:
                return None
            executions += 1
            try:
                output = env.exec_code(
                    _execution_script(
                        code,
                        entry_point,
                        cases,
                        verify=verify,
                        float_rel_tol=self.config["public_float_rel_tol"],
                        float_abs_tol=self.config["public_float_abs_tol"],
                    )
                )
            except Exception:
                return None
            return _execution_result(output, len(cases), verify=verify)

        # Observe only already parsed public verification reports. The
        # inherited execute function, parser, scripts and its cap are unchanged.
        _v111_parent_execute = execute

        def execute(code: str, cases: list[dict], *, verify: bool) -> dict | None:
            result = _v111_parent_execute(code, cases, verify=verify)
            if verify:
                _v111_records.append((code, repr(cases), result))
            return result

        def _v111_ask_residual(user: str) -> str | None:
            nonlocal _v111_extra_calls
            if (
                _v111_extra_calls >= 1
                or _v111_total_limit <= 8
                or env.prompt_cost >= env.prompt_budget
            ):
                return None
            _v111_extra_calls += 1
            try:
                result = env.llm(
                    [
                        {"role": "system", "content": _V111_RESIDUAL_SYSTEM},
                        {"role": "user", "content": user},
                    ]
                )
            except LimitsExceededException:
                return None
            return result if isinstance(result, str) else None

        draft = _code(ask_model(_SOLVE_SYSTEM, original), entry_point)
        question = answer = None
        if env.can_ask():
            diagnoses = _diagnoses(ask_model(_DIAGNOSE_SYSTEM, original), prompt, witness_limit)
            selection_context = (
                original + "\n\nIndependent contract coverage "
                "(proposals, not established truth):\n" + json.dumps(diagnoses, ensure_ascii=True)
            )
            selected = ask_model(_SELECT_SYSTEM, selection_context)
            question = _question(selected, question_limit)
            if question is None:
                retry = ask_model(
                    _QUESTION_FORMAT_SYSTEM,
                    selection_context
                    + "\n\nPrevious proposed question (untrusted):\n"
                    + (selected or "No usable question was produced."),
                )
                question = _question(retry, question_limit)
            question = question or _emergency_question(prompt, entry_point, question_limit)
            # Review only the wording of this already selected target. Keep
            # three later call opportunities for synthesis/applicability/repair;
            # a failed optional review must not suppress the SDK question.
            review_context = (
                selection_context + "\n\nExact candidate question (not a source fact):\n" + question
            )
            if calls + 4 <= call_limit and len(review_context) <= 30000:
                try:
                    review = ask_model(_QUESTION_RECOVERY_SYSTEM, review_context)
                except Exception:
                    review = None
                rewritten = _question_recovery(review, prompt, question, question_limit)
                if rewritten is not None:
                    question = rewritten
            if env.can_ask():
                try:
                    response = env.ask_human(question + "\n\n" + _OUTPUT_FORMAT_REMINDER)
                except (TooManyQuestionException, LimitsExceededException):
                    response = None
                if isinstance(response, str) and response.strip() not in {
                    "",
                    "Irrelevant question.",
                    "Cannot answer the given question.",
                }:
                    answer = response.strip()

        code = draft
        if answer is not None:
            revision_context = (
                original
                + "\n\nClarification question (not itself a fact):\n"
                + str(question)
                + "\n\nActual SDK answer:\n"
                + answer
            )
            revision = ask_model(_REVISE_SYSTEM, revision_context)
            if revision is not None and "KEEP_INITIAL" not in revision:
                code = _code(revision, entry_point) or code

        if code is None:
            recovery_context = (
                original
                + "\n\nActual clarification question (context only):\n"
                + str(question)
                + "\n\nActual usable SDK answer:\n"
                + (
                    answer
                    if answer is not None
                    else "None; do not infer any confirmation or missing fact."
                )
                + "\n\nThe previous response did not yield a syntactically valid "
                "module exposing the public entry. Produce one complete module."
            )
            code = _code(ask_model(_SOLVE_SYSTEM, recovery_context), entry_point)
        if code is None:
            # The mandatory clarification above is still attempted if available.
            code = (
                f"def {entry_point}(*args, **kwargs):\n"
                "    raise NotImplementedError('No valid Python implementation was produced')"
            )
        examples = _public_examples(prompt, answer, entry_point, example_limit)
        if answer is not None:
            numbered = [
                {"index": index, "literal_example": item["evidence"]}
                for index, item in enumerate(examples)
                if item["source"] == "prompt"
            ]
            # With prompt examples this replaces V51's existing applicability
            # call. Without them, use one spare call only when a repair call
            # remains available. A failed normalization is a pure fallback.
            evidence_review = None
            if numbered or calls + 2 <= call_limit:
                evidence_review = ask_model(
                    _EVIDENCE_SYSTEM,
                    original
                    + "\n\nActual question:\n"
                    + str(question)
                    + "\n\nActual SDK answer:\n"
                    + answer
                    + "\n\nOriginal-task literal examples:\n"
                    + json.dumps(numbered, ensure_ascii=True),
                )
            if numbered:
                examples = _applicable_examples(evidence_review, examples)
            witnesses = _answer_witnesses(
                evidence_review, answer, entry_point, answer_witness_limit
            )
            examples = _merge_examples(examples, witnesses, example_limit)
        checked = execute(code, examples, verify=True)
        if checked is not None and _failure_count(checked) > 0:
            repair_context = (
                original
                + "\n\nActual clarification question (context only):\n"
                + str(question)
                + "\n\nActual usable answer:\n"
                + (answer if answer is not None else "None; no new fact was supplied.")
                + "\n\nCurrent candidate:\n"
                + _fenced(code)
                + "\n\nApplicable explicit examples and their visible source spans:\n"
                + repr(examples)
                + "\n\nActual SDK execution records:\n"
                + json.dumps(checked, ensure_ascii=True)
            )
            repaired_response = ask_model(_REPAIR_SYSTEM, repair_context)
            repaired = (
                _code(repaired_response, entry_point)
                if repaired_response is not None and "KEEP_CANDIDATE" not in repaired_response
                else None
            )
            if repaired is not None:
                rechecked = execute(repaired, examples, verify=True)
                if rechecked is not None and _accept_repair(checked, rechecked):
                    code = repaired
        # Every selected candidate must be paired with its own report.
        # Never continue from an unverified, tied or regressing replacement.
        if _v111_records and examples and _v111_total_limit > 8:
            first_code, first_cases, first_report = _v111_records[0]
            same_cases = repr(examples)
            current_report = next(
                (
                    report
                    for candidate, key, report in reversed(_v111_records)
                    if candidate == code and key == same_cases
                ),
                None,
            )
            if (
                code != first_code
                and first_cases == same_cases
                and first_report is not None
                and current_report is not None
                and _accept_repair(first_report, current_report)
                and _failure_count(current_report) > 0
                and executions < exec_limit
                and env.prompt_cost < env.prompt_budget
            ):
                context = (
                    original
                    + "\n\nActual clarification question (context only):\n"
                    + str(question)
                    + "\n\nActual usable SDK answer:\n"
                    + (answer if answer is not None else "None; no new fact was supplied.")
                    + "\n\nCurrent accepted candidate:\n"
                    + _fenced(code)
                    + "\n\nUnchanged admitted public examples and source spans:\n"
                    + repr(examples)
                    + "\n\nActual execution records for this candidate:\n"
                    + json.dumps(current_report, ensure_ascii=True)
                )
                response = _v111_ask_residual(context)
                replacement = (
                    _code(response, entry_point)
                    if response is not None and "KEEP_CANDIDATE" not in response
                    else None
                )
                if replacement is not None:
                    next_report = execute(replacement, examples, verify=True)
                    if next_report is not None and _accept_repair(current_report, next_report):
                        code = replacement

        return _fenced(code)
