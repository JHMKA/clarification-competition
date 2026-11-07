# SPDX-FileCopyrightText: 2026 Pengchao Zheng <pczheng@mail.ustc.edu.cn>
#
# SPDX-License-Identifier: MIT

"""ClariBridge V150.10: bind finite property proposals to exact answer clauses.

Generate candidate implementations and use SDK execution feedback to select a
clarification question. After a reply, compare candidates against grounded
evidence from the exchange and use the remaining budget for bounded repair.

Team: KeepGoing-Z
Team Members: Pengchao Zheng, Mengyi Hu
Main Contact: pczheng@mail.ustc.edu.cn

Algorithm ancestry: project-authored fixed V150, SHA256
2d541adaf9bab7b224c405d7dd3146fecf706b1a203f66cf38b8007a50ce8339.
V150.2 submission parent SHA256:
ebda2c35f5a2c61654eb62800f8facb986b5e91f3c5ba171b58d933a124030f4.
Preserve valid question summaries independently of optional metadata, preserve
target schemas during recovery, and request up to two source-backed semantic
decisions in the existing question call. Choices remain fallible questions;
they never supply executable expected values. Initial construction prompts,
candidate generation, execution, ranking and repair follow the parent.

Direct implementation parent: V150.8, SHA256
500662f5fbc7827dd5206a4c2421ed0439144173c87125a97921e8e24951290f.
Finite property proposals reference controller-owned answer-clause indices;
an exact, unconditional positive return-type contrast is checked locally.
"""

import ast
import concurrent.futures
import copy
import doctest
import hashlib
import io
import json
import math
import pydoc
import re
import threading
import tokenize

from clarify.baselines import ClarificationAlgorithmBase

V129_PROMPTS = {
    "status": "runtime_prompt_inventory_offline_only",
    "language": "English",
    "roles": {
        "initial_builder": "Construct one usable Python implementation for the public operation. Work from original_prompt and entry_point, including the shown constraints and examples. Keep the original wording intact when reasoning about conflicting statements; do not silently delete a requirement. Make reasonable implementation choices without presenting them as facts supplied by the requester. The construction view changes your approach to the same task, not the task itself. Include necessary imports and helper definitions. Respond with just one fenced Python module exposing the requested public callable.",
        "preserving_builder": "You are producing the conservation candidate after an actual clarification. Read original_prompt, question and actual_answer in full, then inspect anchor_source. Retain useful parts of that implementation while changing behavior that the reply actually corrects. For reply_effect=confirm, unchanged code is allowed when it genuinely matches the complete reply. For reply_effect=local_change, apply the scoped correction without spreading it to unrelated inputs. For reply_effect=redefine, keeping the old algorithm is optional: the clarified operation controls. reply_effect is a fallible routing hint, not evidence. Decline no actual obligation merely because an old example disagrees. Return only one fenced Python module with the exact public callable. Consult reply_coverage alongside the full actual_answer. Its retained_reply_clauses and semantic_notes may contain obligations that are not represented by any executable check. Missing checks do not authorize ignoring those obligations. Clause confirmations concern only their identified scope; execution agreement does not confirm the remaining contract.",
        "reconstruction_builder": "Solve the clarified operation anew. Your specification is the unedited original_prompt together with the exact question and actual_answer; do not infer intent from another generated program. Use the reply's explicit scope and preserve remaining requirements. A routing summary or evidence ledger can help organize work, but neither replaces the conversation. Design a coherent algorithm and handle the documented domain rather than patching imaginary tests. Produce a complete implementation as one Python fence and include its public callable. Consult reply_coverage alongside the full actual_answer. Its retained_reply_clauses and semantic_notes may contain obligations that are not represented by any executable check. Missing checks do not authorize ignoring those obligations. Clause confirmations concern only their identified scope; execution agreement does not confirm the remaining contract.",
        "boundary_builder": "Develop an implementation of the same clarified task with particular attention to its documented boundaries, argument roles and result representation. Use the original task and complete real exchange. Different candidate responsibilities are not permission to implement different user requirements. Inspect the supplied scoped obligations for missing boundary cases, while treating speculative or unresolved checks as suggestions. Support the ordinary central behavior as well as the boundaries. Output a single Python-fenced module, including the exact public callable and needed helpers. Consult reply_coverage alongside the full actual_answer. Its retained_reply_clauses and semantic_notes may contain obligations that are not represented by any executable check. Missing checks do not authorize ignoring those obligations. Clause confirmations concern only their identified scope; execution agreement does not confirm the remaining contract.",
        "observation_inputs": "Suggest a compact set of small positional-argument tuples for observing implementations. Supply inputs only; do not guess their outputs. Each observation records argument_literals, domain_evidence, source (prompt or answer) and domain_status (explicit or uncertain). Copy domain evidence from that source when it exists. A vague domain should be marked uncertain rather than manufactured as explicit. After clarification, inspect previous_inputs and the reply before refreshing observations: a usable old input need not be discarded just because you propose new ones. Focus on central behavioral distinctions and permitted boundary cases. Use the observations JSON layout provided in expected_schema.",
        "behavior_question": "Describe the selected implementation as a fallible behavior contract for its requester to correct. Supply input_summary (argument roles and shapes), result_summary (return representation), and operation_summary (the central mapping, including ordinary cases). These describe anchor_source, not established requirements. Do not write a question, invent example outputs, or quote implementation text. Select witness_id from anchor_observations, preferring a normal valid input that separates candidates; the controller will print its recorded values. Set conflict_spans to two short exact original passages only when their core requirements cannot both hold, otherwise []. Keep each summary concise and declarative. A local ambiguity must not displace coverage of the complete central operation. Use the supplied JSON schema. Also identify up to two unresolved behavior decisions from original_prompt, even when all drafts agree. Prefer the central mapping or a contradictory requirement before incidental boundaries. For each decision copy source_span exactly from original_prompt; name dimension and status; give a concrete missing detail, two plausible alternative rules and how the rules change behavior. Do not invent an ambiguity already settled by the prompt. Both alternatives are hypotheses, not requirements or an exhaustive list. Do not invent input/output values or use task familiarity as a specification. Use decisions=[] when there is no defensible unresolved decision. Keep all three contract summaries useful even if witness_id is empty or conflict_spans is []. Choose witness_id only from allowed_witness_ids, never from other measurement IDs.",
        "question_revision": "Complete the missing or malformed parts of the behavior contract record using anchor_source and the public description. Preserve uncertainty instead of supplying an imagined requirement. Include input_summary, result_summary, operation_summary and a supplied witness_id when one exists. The controller owns the final request and the execution example; do not compose either. conflict_spans must be empty or a supported pair of original passages. Return the requested JSON record. Also identify up to two unresolved behavior decisions from original_prompt, even when all drafts agree. Prefer the central mapping or a contradictory requirement before incidental boundaries. For each decision copy source_span exactly from original_prompt; name dimension and status; give a concrete missing detail, two plausible alternative rules and how the rules change behavior. Do not invent an ambiguity already settled by the prompt. Both alternatives are hypotheses, not requirements or an exhaustive list. Do not invent input/output values or use task familiarity as a specification. Use decisions=[] when there is no defensible unresolved decision. Keep all three contract summaries useful even if witness_id is empty or conflict_spans is []. Choose witness_id only from allowed_witness_ids, never from other measurement IDs.",
        "reply_map": "Organize the exact actual_answer for the controller while keeping the entire conversation authoritative. Set effect to confirm, local_change, redefine or unclear, and quote the reply passage behind effect_span; this label only routes construction. Record a short core_intent summary without adding new requirements. call_results contains only input/output values directly written in the reply: argument_literals is a positional tuple, expected_literal is the stated value, and source_span is its full supporting passage. Do not turn a model-derived answer into a literal fact. A plain yes cannot resolve an either/or question. For properties use only the declared finite node kinds with their actual subject, scope and conditions. An input described as a string does not make the return value a string. Preserve negation and the condition that governs a claim. Suggest example_dispositions only where the real reply covers the old example; suggest input_updates only where the reply establishes that input's validity or exclusion. Do not remove an example because generated programs fail it. Return the declared JSON object, using empty lists for unavailable evidence. Conditional properties use arg_compare {kind,index,op,value}, arg_empty {kind,index,empty}, or arg_type {kind,index,value}. Indices are zero-based. Comparison op is eq, ne, lt, le, gt or ge; comparison value is a literal string. Omit condition for unconditional claims. Use bool, int, float, str, list, tuple, dict or set for return_type, and ascending or descending for sorted_order. The question_binding carries separately identified input, result, operation and measured-example clauses. Populate clause_responses with clause_id, verdict (confirmed, corrected, unresolved), and the FULL governing answer sentence as source_span. A general yes to the compound contract does not confirm every clause or the example; leave those unresolved unless the reply identifies their scope. Example confirmation must actually refer to the shown example or its clause ID, with no negation or qualification. Never turn confirmation of representation into confirmation of a numeric result. An explicit corrected input/output pair has precedence over a confirmation. Unexpressed details remain unknown. You may list semantic_notes with a verbatim source_span and a fallible interpretation for requirements that the finite checker cannot express; those notes are for builders and review, not executable oracles.",
        "evidence_arbitration": "Inspect the listed evidence conflict using source text and the real clarification, not popularity among programs. This is a single local review, not a rewrite of the requirement. For each supplied id choose retain, supersede or uncertain, and provide source_span from the actual reply, original_span when relevant, scope and reason. A superseding decision needs a reply statement that applies to the same case or requirement. Repeated execution failure is a reason to investigate; it is not proof that a check is wrong. Distinguish a faulty expected value, an inapplicable input, a program defect and unavailable measurements. If the source does not settle the conflict, keep that uncertainty explicit. Emit only the decisions JSON object.",
        "obligation_rescue": "Build one additional answer-conditioned candidate because the current set has a grounded deficit. Read the original task, exact exchange, active obligations and measured artifacts. Resolve failures supported by those sources without altering the checks to fit your code. Missing observations are not evidence of a wrong program. A suggested route or inferred rule remains fallible. You may choose a new algorithm when the clarified behavior requires it. Supply the entire Python implementation in one code fence with the required public callable. Consult reply_coverage alongside the full actual_answer. Its retained_reply_clauses and semantic_notes may contain obligations that are not represented by any executable check. Missing checks do not authorize ignoring those obligations. Clause confirmations concern only their identified scope; execution agreement does not confirm the remaining contract.",
        "execution_patch": "Revise the current_source to address a trustworthy runtime mismatch. Use the complete original task and actual clarification, then separate the intended behavior from the recorded input, observed output, exception and expected-value source. Input-domain uncertainty or a guessed expectation is not enough to declare the implementation faulty. Preserve already established answer obligations and applicable behavior outside the correction's scope. If the error concerns an allowed input with no exact expected value, repair the real exception without inventing a new output rule. Keep the change focused unless the specification requires a different algorithm. Respond only with the full revised module in one Python fence. Consult reply_coverage alongside the full actual_answer. Its retained_reply_clauses and semantic_notes may contain obligations that are not represented by any executable check. Missing checks do not authorize ignoring those obligations. Clause confirmations concern only their identified scope; execution agreement does not confirm the remaining contract.",
        "reply_compliance_review": "Check whether the selected implementation has a concrete behavioral conflict with the original task or actual_answer. Treat candidate consensus, effect labels and missing measurements as fallible context. For a supported conflict, return decision=propose with the verbatim source_span, a concise reason and the complete replacement_source. Otherwise return decision=retain, explaining the relevant uncertainty briefly. A stylistic preference or hypothetical requirement cannot justify replacement. Any proposed module will be executed against the same active evidence; do not claim that your judgment itself establishes correctness. Return JSON only. Consult reply_coverage alongside the full actual_answer. Its retained_reply_clauses and semantic_notes may contain obligations that are not represented by any executable check. Missing checks do not authorize ignoring those obligations. Clause confirmations concern only their identified scope; execution agreement does not confirm the remaining contract.",
        "schema_recovery": "Re-encode the previous_payload in the requested output_kind for target_role. Preserve its substantive algorithm, facts, conditions and uncertainty. For a module, emit a single Python code fence; for a record, use the declared JSON field types and represent absent information with empty or unknown values. Do not solve the task again or introduce missing behavior merely to fill a field. Return the repaired representation by itself.",
    },
    "user_envelope": "TASK_PACKET:\n{packet_json}",
    "sdk_question_suffix": "Response-format reminder: follow the Output Format in your system instructions. Write ANSWERS=`<answer_to_question>` and keep backticks outside the answer body. Preserve the factual answer unchanged.",
    "construction_views": {
        "initial": [
            "derive the main mapping from the documented input and result contract",
            "organize a separate solution around the task's defining invariant",
            "consider permitted boundary cases alongside ordinary behavior",
        ]
    },
    "schemas": {
        "observation_inputs": {
            "observations": [
                {
                    "argument_literals": "positional tuple literal",
                    "domain_evidence": "verbatim source excerpt when available",
                    "source": "prompt|answer",
                    "domain_status": "explicit|uncertain",
                }
            ]
        },
        "behavior_question": {
            "input_summary": "short description of draft arguments, not a requirement",
            "result_summary": "draft return type and structure",
            "operation_summary": "draft central input-to-output operation",
            "witness_id": "existing anchor observation ID, or empty",
            "conflict_spans": ["two literal core-conflict passages, or empty list"],
            "decisions": [
                {
                    "dimension": "operation|result|input|boundary|conflict",
                    "status": "unresolved|contradictory",
                    "source_span": "short exact original_prompt excerpt",
                    "detail": "specific semantic decision, not a general request for clarification",
                    "alternatives": ["possible rule A", "possible rule B"],
                    "behavior_effect": "concrete behavioral consequence of this decision",
                }
            ],
        },
        "question_revision": {
            "input_summary": "short description of draft arguments, not a requirement",
            "result_summary": "draft return type and structure",
            "operation_summary": "draft central input-to-output operation",
            "witness_id": "existing anchor observation ID, or empty",
            "conflict_spans": ["two literal core-conflict passages, or empty list"],
            "decisions": [
                {
                    "dimension": "operation|result|input|boundary|conflict",
                    "status": "unresolved|contradictory",
                    "source_span": "short exact original_prompt excerpt",
                    "detail": "specific semantic decision, not a general request for clarification",
                    "alternatives": ["possible rule A", "possible rule B"],
                    "behavior_effect": "concrete behavioral consequence of this decision",
                }
            ],
        },
        "reply_map": {
            "effect": "confirm|local_change|redefine|unclear",
            "effect_span": "verbatim actual reply passage",
            "core_intent": "short fallible summary",
            "call_results": [
                {
                    "argument_literals": "positional tuple literal",
                    "expected_literal": "value literally stated in reply",
                    "source_span": "reply passage",
                }
            ],
            "properties": [],
            "example_dispositions": [
                {
                    "id": "existing example ID",
                    "status": "superseded|unresolved",
                    "response_span": "reply passage",
                    "original_span": "original example passage",
                    "covered_condition": "scope shared by the statement and example",
                }
            ],
            "input_updates": [
                {
                    "id": "existing input ID",
                    "action": "keep|exclude|uncertain",
                    "source_span": "actual reply passage",
                    "reason": "domain effect",
                }
            ],
            "clause_responses": [
                {
                    "clause_id": "I0|O0|R0|E0|D0|D1",
                    "verdict": "confirmed|corrected|unresolved",
                    "source_span": "whole governing answer sentence",
                }
            ],
            "semantic_notes": [
                {
                    "source_span": "verbatim governing reply clause",
                    "interpretation": "fallible natural-language consequence",
                }
            ],
        },
        "evidence_arbitration": {
            "decisions": [
                {
                    "id": "supplied disputed record ID",
                    "action": "retain|supersede|uncertain",
                    "source_span": "verbatim actual reply passage",
                    "original_span": "original evidence passage when relevant",
                    "scope": "covered input or requirement scope",
                    "reason": "source-based explanation, not passing counts",
                }
            ]
        },
        "reply_compliance_review": {
            "decision": "retain|propose",
            "source_span": "supporting original or reply passage when proposing",
            "replacement_source": "complete module only when proposing",
            "reason": "specific conflict or uncertainty",
        },
    },
    "provenance": "V129 is an independently implemented extension of project V128.",
}

V129_PROMPTS["roles"]["reply_map"] = (
    V129_PROMPTS["roles"]["reply_map"]
    .replace(
        "For properties use only the declared finite node kinds with their actual subject, scope and conditions. "
        "An input described as a string does not make the return value a string. "
        "Preserve negation and the condition that governs a claim. ",
        "For properties, follow property_node_contract and cite only an allowed "
        "answer_property_sources index. Retain scope and conditions; an input, draft "
        "or excluded type is not the returned type. The controller supplies source "
        "text; emit no source_span/scope_clause. Use [] for unsupported or unproved nodes. ",
    )
    .replace(
        "Conditional properties use arg_compare {kind,index,op,value}, arg_empty {kind,index,empty}, "
        "or arg_type {kind,index,value}. Indices are zero-based. Comparison op is eq, ne, lt, le, gt or ge; "
        "comparison value is a literal string. Omit condition for unconditional claims. "
        "Use bool, int, float, str, list, tuple, dict or set for return_type, and ascending or descending "
        "for sorted_order. ",
        "",
    )
)


V129_PROMPTS["roles"].update(
    {
        "candidate_review": 'Assess the supplied implementations against the original request and the complete actual clarification. Execution receipts already decide explicit checks; do not invent expected outputs or use the number of agreeing programs as evidence of intent. Inspect central computation, argument meaning, return shape, negation and exceptions stated in the conversation. Select one supplied candidate_index only if a specific source passage supports that choice over a competing implementation. Quote that exact passage as source_span and give a short concrete reason. If the conversation cannot distinguish them, return candidate_index=-1, source_span="", reason="unresolved". A suggested plan and another reviewer are fallible context. Do not supply new code. Return the declared JSON record.',
        "implementation_plan": "Prepare at most three different implementation routes for the exact clarified operation. Use the entire original request and actual conversation. Each route needs a short exact source_span and a concise algorithm outline. Distinguish a permitted implementation choice from an asserted user requirement. Never manufacture test answers, hidden cases, or missing constraints. Plans organize candidate construction and have no authority over the original text. Return only the routes JSON.",
    }
)
V129_PROMPTS["schemas"].update(
    {
        "candidate_review": {
            "candidate_index": "zero-based supplied index, or -1",
            "source_span": "exact supporting conversation passage",
            "reason": "specific source-grounded reason",
        },
        "implementation_plan": {
            "routes": [
                {
                    "source_span": "exact conversation passage",
                    "outline": "fallible implementation route",
                }
            ]
        },
    }
)


def _campaign_packet(role, packet, policy):
    packet = dict(packet)
    if policy.get("raw_reconstruction") and role == "reconstruction_builder":
        packet = {
            key: packet[key]
            for key in ("original_prompt", "entry_point", "question", "actual_answer")
            if key in packet
        }
    if policy.get("literal_clauses") and role in {
        "preserving_builder",
        "reconstruction_builder",
        "boundary_builder",
        "execution_patch",
        "obligation_rescue",
    }:
        text = packet.get("actual_answer", "")
        # Verbatim indexing adds no interpretation or executable oracle.
        packet["literal_answer_clauses"] = [
            part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()
        ][:40]
    return packet


def _campaign_disagreement(report, inputs):
    live = [item for item in inputs if item.get("domain_status") != "excluded"]
    for index, item in enumerate(live):
        if item.get("domain_status") != "confirmed":
            continue
        values = {
            row["observations"][index].get("result_key")
            for row in report["candidates"]
            if index < len(row.get("observations", []))
            and row["observations"][index].get("ok") is True
            and row["observations"][index].get("result_key")
        }
        if len(values) > 1:
            return True
    return False


def _read_literal(source):
    if not isinstance(source, str) or len(source) > 12000:
        raise ValueError("literal size")
    node = ast.parse(source, mode="eval")
    if sum(1 for _ in ast.walk(node)) > 300:
        raise ValueError("literal complexity")
    return ast.literal_eval(node)


def _entry_present(module, entry):
    known = set()
    for statement in module.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            known.add(statement.name)
        elif (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            if (
                isinstance(statement.value, ast.Lambda)
                or isinstance(statement.value, ast.Name)
                and statement.value.id in known
            ):
                known.add(statement.targets[0].id)
    return entry in known


def extract_code(response, entry_point, declared=None):
    if not isinstance(response, str) or len(response) > 100000:
        return None
    blocks = re.findall(r"```(?:[Pp]ython|py)?\s*\n([\s\S]*?)```", response)
    for item in blocks + [response]:
        try:
            module = ast.parse(item.strip())
            if _entry_present(module, entry_point):
                return item.strip()
        except (SyntaxError, ValueError, RecursionError):
            continue
    return None


def sanitize(source):
    module = ast.parse(source)
    permitted = []
    for statement in module.body:
        if isinstance(
            statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Import, ast.ImportFrom)
        ):
            permitted.append(statement)
        elif (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        ):
            permitted.append(statement)
        elif isinstance(statement, (ast.Assign, ast.AnnAssign)):
            # Keep computed module constants and legitimate initialization intact.
            # Candidate execution is confined to the SDK sandbox, never this parser.
            permitted.append(statement)
    module.body = permitted
    return ast.unparse(module)


def declared_name(prompt, entry):
    declarations = set(re.findall(r"(?m)^\s*def\s+([A-Za-z_]\w*)\s*\(", prompt))
    return (
        next(iter(declarations)) if len(declarations) == 1 and entry not in declarations else None
    )


def _group_end(text, start):
    level = 0
    quote = None
    escaped = False
    for position in range(start, len(text)):
        char = text[position]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in "'\"":
            quote = char
        elif char in "([{":
            level += 1
        elif char in ")]}":
            level -= 1
            if level == 0:
                return position + 1
            if level < 0:
                return None
    return None


def _prefix_value(text):
    text = text.lstrip()
    if not text:
        raise ValueError("no value")
    if text[0] in "([{":
        end = _group_end(text, 0)
        if end is None:
            raise ValueError("unclosed literal")
        return _read_literal(text[:end])
    tokens = list(tokenize.generate_tokens(io.StringIO(text.splitlines()[0]).readline))
    meaningful = [
        t for t in tokens if t.type not in {tokenize.ENDMARKER, tokenize.NEWLINE, tokenize.NL}
    ]
    if not meaningful:
        raise ValueError("no token")
    end = meaningful[0].end[1]
    if meaningful[0].string in {"+", "-"} and len(meaningful) > 1:
        end = meaningful[1].end[1]
    token_text = text[:end]
    if re.fullmatch(r"[+-]?\d+\.", token_text):
        token_text = token_text[:-1]
    return _read_literal(token_text)


def _call_literal(node, names):
    if (
        not isinstance(node, ast.Call)
        or not isinstance(node.func, ast.Name)
        or node.func.id not in names
        or node.keywords
    ):
        return None
    try:
        return tuple(_read_literal(ast.unparse(arg)) for arg in node.args)
    except (ValueError, TypeError, SyntaxError, RecursionError):
        return None


def _typed_equal(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(_typed_equal(x, y) for x, y in zip(a, b, strict=False))
    if isinstance(a, dict):
        return len(a) == len(b) and all(
            any(_typed_equal(k, other_key) and _typed_equal(v, w) for other_key, w in b.items())
            for k, v in a.items()
        )
    return a == b


def answer_relations(answer, names):
    if not isinstance(answer, str):
        return []
    found = []
    for name in names:
        if not isinstance(name, str):
            continue
        for match in re.finditer(r"\b" + re.escape(name) + r"\s*\(", answer):
            prefix = answer[max(0, match.start() - 60) : match.start()]
            if re.search(r"\b(?:not|never|instead of|rather than)\s*$", prefix, re.I):
                continue
            end = _group_end(answer, answer.index("(", match.start()))
            if end is None:
                continue
            try:
                node = ast.parse(answer[match.start() : end], mode="eval").body
                args = _call_literal(node, names)
                if args is None:
                    continue
                tail = answer[end:]
                relation = re.match(
                    r"\s*(?:==|->|=>|should\s+return|must\s+return|returns?|outputs?)\s*",
                    tail,
                    re.I,
                )
                if relation:
                    expected = _prefix_value(tail[relation.end() :])
                    if not stated_in(found, args, expected):
                        found.append((args, expected))
            except (ValueError, TypeError, SyntaxError, tokenize.TokenError, RecursionError):
                continue
    return found


def stated_in(relations, args, expected):
    return any(
        _typed_equal(tuple(args), tuple(a)) and _typed_equal(expected, b) for a, b in relations
    )


def parse_examples(prompt, names):
    results = []
    try:
        examples = doctest.DocTestParser().get_examples(prompt)
    except ValueError:
        examples = []
    for example in examples:
        try:
            expression = ast.parse(example.source.strip(), mode="eval").body
            args = _call_literal(expression, names)
            expected = _read_literal(example.want.strip())
            if args is not None and not stated_in(results, args, expected):
                results.append((args, expected))
        except (ValueError, TypeError, SyntaxError, RecursionError):
            continue
    for line in prompt.splitlines():
        try:
            statement = ast.parse(line.strip()).body[0]
            expression = (
                statement.test
                if isinstance(statement, ast.Assert)
                else statement.value
                if isinstance(statement, ast.Expr)
                else None
            )
            if (
                isinstance(expression, ast.Compare)
                and len(expression.ops) == 1
                and isinstance(expression.ops[0], ast.Eq)
            ):
                args = _call_literal(expression.left, names)
                expected = _read_literal(ast.unparse(expression.comparators[0]))
                if args is not None and not stated_in(results, args, expected):
                    results.append((args, expected))
        except (ValueError, TypeError, SyntaxError, IndexError, RecursionError):
            continue
    return results[:12]


def clean_question(text):
    if not isinstance(text, str):
        return ""
    normalized = text.strip()
    normalized = re.sub(r"^\s*(?:ASK|Q|question)\s*:\s*", "", normalized, flags=re.I)
    if normalized.startswith("```"):
        normalized = "\n".join(normalized.splitlines()[1:-1])
    return " ".join(normalized.split())


def call_text(entry, args_repr):
    try:
        args = _read_literal(args_repr) if isinstance(args_repr, str) else args_repr
        if isinstance(args, tuple):
            return entry + "(" + ", ".join(map(repr, args)) + ")"
    except (ValueError, SyntaxError, TypeError, RecursionError):
        pass
    return entry + "(" + str(args_repr) + ")"


INDEPENDENT_EXECUTION_PROGRAM = r"""
import copy
import json
import math
import signal
import time
import hashlib
import io
import contextlib


# Exact structural keys and approximate check comparisons serve different purposes.
# Numeric values use absolute 1e-6 and relative 1e-7 tolerance.
# Explicit type obligations are evaluated separately from value equality.
_FLOAT_ABS_TOLERANCE = 1e-6


class CallTimeout(BaseException):
    pass


class BudgetExhausted(BaseException):
    pass


class EntryPointError(Exception):
    pass


class _CapturedOutput(io.StringIO):
    def write(self, value):
        count = len(value)
        available = max(0, 8192 - self.tell())
        if available:
            super().write(value[:available])
        return count


def _error(exc, skipped=False):
    frames = []
    tb = exc.__traceback__
    while tb is not None:
        code = tb.tb_frame.f_code
        if code.co_filename.startswith('<candidate_'):
            frames.append({'file': code.co_filename,
                           'function': code.co_name,
                           'line': tb.tb_lineno})
        tb = tb.tb_next
    record = {'type': type(exc).__name__, 'message': str(exc)[:2000]}
    if frames:
        record['trace'] = frames[-8:]
    if skipped:
        record['skipped'] = True
    return record


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False)


def _typed_tree(value, active=None):
    if active is None:
        active = set()
    kind = type(value)
    if value is None:
        return ['none']
    if kind is bool:
        return ['bool', value]
    if kind is int:
        return ['int', str(value)]
    if kind is float:
        return ['float', repr(value)]
    if kind is str:
        return ['str', value]
    if kind is bytes:
        return ['bytes', value.hex()]
    if kind not in (list, tuple, dict, set, frozenset):
        raise TypeError('Unsupported result type: ' + kind.__name__)
    identity = id(value)
    if identity in active:
        raise ValueError('Cyclic values cannot be structurally fingerprinted')
    active.add(identity)
    try:
        if kind in (list, tuple):
            tag = 'list' if kind is list else 'tuple'
            return [tag, [_typed_tree(item, active) for item in value]]
        if kind is dict:
            pairs = [[_typed_tree(key, active), _typed_tree(item, active)]
                     for key, item in value.items()]
            pairs.sort(key=_canonical)
            return ['dict', pairs]
        items = [_typed_tree(item, active) for item in value]
        items.sort(key=_canonical)
        return ['set' if kind is set else 'frozenset', items]
    finally:
        active.remove(identity)


def _match_unordered(left, right, compatible):
    # A complete matching avoids order-dependent decisions for close float keys.
    options = [[j for j, other in enumerate(right) if compatible(item, other)]
               for item in left]
    if any(not row for row in options):
        return False
    order = sorted(range(len(left)), key=lambda index: len(options[index]))

    def visit(position, used):
        if position == len(order):
            return True
        for target in options[order[position]]:
            if target not in used:
                used.add(target)
                if visit(position + 1, used):
                    return True
                used.remove(target)
        return False

    return visit(0, set())


def _same_value(left, right):
    numeric = (bool, int, float)
    if type(left) in numeric and type(right) in numeric:
        if type(left) is not float and type(right) is not float:
            return left == right
        if left == right:
            return True
        try:
            return math.isclose(left, right, rel_tol=1e-7,
                                abs_tol=_FLOAT_ABS_TOLERANCE)
        except (OverflowError, TypeError, ValueError):
            return False
    kind = type(left)
    if kind in (set, frozenset) and type(right) in (set, frozenset):
        return len(left) == len(right) and _match_unordered(
            list(left), list(right), _same_value)
    if kind is not type(right):
        return False
    if left is None or kind in (str, bytes):
        return left == right
    if kind in (list, tuple):
        return len(left) == len(right) and all(
            _same_value(a, b) for a, b in zip(left, right))
    if kind is dict:
        return len(left) == len(right) and _match_unordered(
            list(left.items()), list(right.items()),
            lambda a, b: _same_value(a[0], b[0]) and _same_value(a[1], b[1]))
    raise TypeError('Unsupported comparison type: ' + kind.__name__)


def _timed(action):
    remaining = _deadline - time.monotonic()
    if remaining <= 0:
        raise BudgetExhausted('The shared execution budget is exhausted')
    budget_limited = remaining <= _per_call_seconds
    allowance = min(remaining, _per_call_seconds)

    def alarm_handler(signum, frame):
        if budget_limited:
            raise BudgetExhausted('The shared execution budget expired during execution')
        raise CallTimeout('The per-call execution limit was exceeded')

    prior_handler = signal.getsignal(signal.SIGALRM)
    prior_timer = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, allowance)
    try:
        return action()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prior_handler)
        if prior_timer[0] > 0:
            prior_remaining = max(1e-9, prior_timer[0] - (time.monotonic() - started))
            signal.setitimer(signal.ITIMER_REAL, prior_remaining, prior_timer[1])


def _skipped(reason):
    return {'type': 'Skipped', 'message': reason, 'skipped': True}


def _check_skipped(reason):
    return {'ok': None, 'actual_repr': None, 'error': _skipped(reason)}


def _observation_skipped(args, reason):
    return {'ok': None, 'result_key': None, 'result_repr': None,
            'error': _skipped(reason),
            'mutated_arguments': {'before': repr(args), 'after': None, 'changed': None}}



_PROPERTY_TYPES = {'bool': bool, 'int': int, 'float': float, 'str': str,
                   'list': list, 'tuple': tuple, 'dict': dict, 'set': set}


def _argument(args, index):
    if type(index) is not int or not 0 <= index < len(args):
        raise ValueError('Invalid property argument index')
    return args[index]


def _condition_applies(condition, args):
    if condition is None:
        return True
    if type(condition) is not dict:
        raise ValueError('A condition must be a declared condition node')
    kind = condition.get('kind')
    value = _argument(args, condition.get('index'))
    if kind == 'arg_empty':
        if type(condition.get('empty')) is not bool:
            raise ValueError('arg_empty needs a boolean empty field')
        if type(value) not in (list, tuple, str, bytes, dict, set, frozenset):
            raise TypeError('Empty condition applies to a supported container')
        return (len(value) == 0) == condition['empty']
    if kind == 'arg_type':
        expected = _PROPERTY_TYPES.get(condition.get('value'))
        if expected is None:
            raise ValueError('Unsupported condition type')
        return type(value) is expected
    if kind == 'arg_compare':
        op = condition.get('op')
        expected = condition.get('value')
        _typed_tree(expected)
        if op == 'eq':
            return bool(value == expected)
        if op == 'ne':
            return bool(value != expected)
        if op == 'lt':
            return bool(value < expected)
        if op == 'le':
            return bool(value <= expected)
        if op == 'gt':
            return bool(value > expected)
        if op == 'ge':
            return bool(value >= expected)
        raise ValueError('Unsupported comparison condition')
    raise ValueError('Unsupported condition kind')


def _property_holds(prop, args, result):
    if type(prop) is not dict or not isinstance(prop.get('id'), str):
        raise ValueError('A property must be a named property node')
    kind = prop.get('kind')
    if kind == 'return_type':
        expected = _PROPERTY_TYPES.get(prop.get('value'))
        if expected is None:
            raise ValueError('Unsupported return type')
        return type(result) is expected
    if kind == 'length_equals_input':
        source = _argument(args, prop.get('arg_index'))
        return len(result) == len(source)
    if kind == 'members_from_input':
        source = _argument(args, prop.get('arg_index'))
        if type(result) not in (list, tuple, str, bytes, dict, set, frozenset):
            raise TypeError('Membership result must be a supported container')
        if type(source) not in (list, tuple, str, bytes, dict, set, frozenset):
            raise TypeError('Membership source must be a supported container')
        return all(any(item == original for original in source) for item in result)
    if kind == 'sorted_order':
        if prop.get('value') not in ('ascending', 'descending'):
            raise ValueError('Unsupported sorting direction')
        if type(result) not in (list, tuple, str, bytes):
            raise TypeError('Ordering result must be an ordered container')
        pairs = zip(result, result[1:])
        if prop['value'] == 'ascending':
            return all(a <= b for a, b in pairs)
        return all(a >= b for a, b in pairs)
    if kind == 'unique_elements':
        if type(result) not in (list, tuple, str, bytes, dict, set, frozenset):
            raise TypeError('Unique result must be a supported container')
        items = list(result)
        return all(not any(item == previous for previous in items[:i])
                   for i, item in enumerate(items))
    raise ValueError('Unsupported property kind')


def _property_observation(prop, supplied_args, result_available, result):
    try:
        applicable = _condition_applies(prop.get('condition'), supplied_args)
        if not applicable:
            return {'id': prop.get('id'), 'ok': None, 'applicable': False,
                    'error': None}
        if not result_available:
            return {'id': prop.get('id'), 'ok': None, 'applicable': True,
                    'error': {'type': 'ResultUnavailable',
                              'message': 'No completed candidate result is available'}}
        checked = bool(_property_holds(prop, supplied_args, result))
        return {'id': prop.get('id'), 'ok': checked, 'applicable': True,
                'error': None}
    except (CallTimeout, BudgetExhausted):
        raise
    except BaseException as exc:
        return {'id': prop.get('id') if isinstance(prop, dict) else None,
                'ok': None, 'applicable': None,
                'error': _error(exc, skipped=isinstance(exc, BudgetExhausted))}


def _aggregate_property(prop, observations):
    cases = [event for observation in observations
             for event in observation.get('properties', [])
             if event['id'] == prop.get('id')]
    applicable = [event for event in cases if event['applicable'] is not False]
    covered = sum(type(event['ok']) is bool for event in applicable)
    failures = [event for event in applicable if event['ok'] is False]
    unknown = [event for event in applicable if event['ok'] is None]
    ok = False if failures else None if unknown or covered == 0 else True
    errors = [event['error'] for event in applicable if event.get('error')]
    error = None if failures else errors[0] if errors else (
        {'type': 'NoApplicableInput', 'message': 'No applicable supplied probe was evaluated'}
        if covered == 0 else None)
    return {'id': prop.get('id'), 'ok': ok, 'covered': covered,
            'error': error}

def _invoke(entrypoint, supplied_args, expected, checking, properties=()):
    state = {'arguments': None, 'before': None, 'before_key': None,
             'after': None, 'changed': None, 'actual_repr': None,
             'result_key': None, 'matches': None, 'property_events': []}

    def work():
        if type(supplied_args) is not tuple:
            raise TypeError('Call arguments must be supplied as a tuple')
        arguments = copy.deepcopy(supplied_args)
        state['arguments'] = arguments
        state['before'] = repr(arguments)
        state['before_key'] = _canonical(_typed_tree(arguments))
        result = entrypoint(*arguments)
        state['actual_repr'] = repr(result)
        result_tree = _typed_tree(result)
        state['result_key'] = hashlib.sha256(
            _canonical(result_tree).encode('utf-8')).hexdigest()
        if not checking:
            state['property_events'] = [
                _property_observation(prop, supplied_args, True, result)
                for prop in properties]
        if checking:
            # Validate the expected structure before recursively comparing it.
            _typed_tree(expected)
            state['matches'] = _same_value(result, expected)
        state['after'] = repr(arguments)
        state['changed'] = state['before_key'] != _canonical(_typed_tree(arguments))

    failure = None
    with contextlib.redirect_stdout(_CapturedOutput()), contextlib.redirect_stderr(_CapturedOutput()):
        try:
            _timed(work)
        except BaseException as exc:
            failure = _error(exc, skipped=isinstance(exc, BudgetExhausted))
            # On ordinary exceptions, preserve mutation evidence under a fresh,
            # bounded operation. Timed-out calls never get an unbounded cleanup.
            if state['arguments'] is not None and not isinstance(exc, (CallTimeout, BudgetExhausted)):
                def snapshot():
                    state['after'] = repr(state['arguments'])
                    state['changed'] = state['before_key'] != _canonical(
                        _typed_tree(state['arguments']))
                try:
                    _timed(snapshot)
                except BaseException:
                    pass
    status = None if failure and failure.get('skipped') else (
        False if failure else (state['matches'] if checking else True))
    if checking:
        return {'ok': status, 'actual_repr': state['actual_repr'], 'error': failure}
    if not state['property_events']:
        state['property_events'] = [
            _property_observation(prop, supplied_args, False, None)
            for prop in properties]
    return {'ok': status, 'result_key': state['result_key'],
            'result_repr': state['actual_repr'], 'error': failure,
            'properties': state['property_events'],
            'mutated_arguments': {'before': state['before'],
                                  'after': state['after'],
                                  'changed': state['changed']}}


def _run(request):
    global _deadline, _per_call_seconds
    if type(request) is not dict:
        raise TypeError('EXECUTION_REQUEST must be a dictionary')
    name = request['entry']
    sources = request['candidates']
    checks = request.get('checks', [])
    inputs = request.get('inputs', [])
    properties = request.get('properties', [])
    if type(properties) is not list or any(type(prop) is not dict for prop in properties):
        raise TypeError('properties must be a list of declared nodes')
    ids = [prop.get('id') for prop in properties]
    if any(type(ident) is not str or not ident for ident in ids) or len(set(ids)) != len(ids):
        raise ValueError('properties require unique string ids')
    if type(name) is not str or not name.isidentifier() or name.startswith('_'):
        raise ValueError('entry must name a public Python identifier')
    if type(sources) is not list or any(type(source) is not str for source in sources):
        raise TypeError('candidates must be a list of source strings')
    for field in ('per_call_seconds', 'total_seconds'):
        value = request.get(field, 2 if field == 'per_call_seconds' else 25)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(field + ' must be a positive finite number')
    _per_call_seconds = request.get('per_call_seconds', 2)
    _deadline = time.monotonic() + request.get('total_seconds', 25)
    ledger = {'candidates': [], 'input_literals': [repr(args) for args in inputs],
              'float_tolerance': {'absolute': _FLOAT_ABS_TOLERANCE, 'relative': 1e-7},
              'result_key_format': 'sha256 of canonical typed structure; floats are exact'}
    for index, source in enumerate(sources):
        namespace = {'__name__': '__candidate__'}
        record = {'load_error': None, 'checks': [], 'observations': [], 'properties': []}
        ledger['candidates'].append(record)

        def load():
            exec(compile(source, '<candidate_' + str(index) + '>', 'exec'), namespace)
            if name not in namespace or not callable(namespace[name]):
                raise EntryPointError('The exact public entry is missing or is not callable: ' + name)
            return namespace[name]

        with contextlib.redirect_stdout(_CapturedOutput()), contextlib.redirect_stderr(_CapturedOutput()):
            try:
                entrypoint = _timed(load)
            except BaseException as exc:
                record['load_error'] = _error(exc, skipped=isinstance(exc, BudgetExhausted))
        if record['load_error'] is not None:
            reason = 'Candidate entry was unavailable because loading did not complete'
            record['checks'] = [_check_skipped(reason) for _ in checks]
            record['observations'] = [_observation_skipped(args, reason) for args in inputs]
            record['properties'] = [dict(id=prop['id'], ok=None, covered=0,
                                          error=_skipped(reason)) for prop in properties]
            continue
        for args, expected in checks:
            record['checks'].append(_invoke(entrypoint, args, expected, True))
        for args in inputs:
            record['observations'].append(_invoke(entrypoint, args, None, False, properties))
        record['properties'] = [_aggregate_property(prop, record['observations'])
                                for prop in properties]
    return ledger


try:
    _ledger = _run(EXECUTION_REQUEST)
except BaseException as _exc:
    _ledger = {'candidates': [], 'input_literals': [],
               'configuration_error': _error(_exc),
               'float_tolerance': {'absolute': _FLOAT_ABS_TOLERANCE, 'relative': 1e-7}}
print('@@EVIDENCE_LEDGER@@' + json.dumps(_ledger, sort_keys=True, ensure_ascii=True, allow_nan=False))
"""


"""V129 own evidence helpers, based on project-authored V127 parsing. No executable model predicates."""


def _v129_quote_at(text, i):
    char = text[i]
    if char not in "'\"":
        return None
    if char == "'" and i and i + 1 < len(text) and text[i - 1].isalnum() and text[i + 1].isalnum():
        return None
    return char * 3 if text.startswith(char * 3, i) else char


def _v129_clauses(answer):
    if not isinstance(answer, str):
        return []
    out = []
    start = 0
    stack = []
    quote = None
    escape = False
    i = 0
    while i < len(answer):
        char = answer[i]
        if quote:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif answer.startswith(quote, i):
                i += len(quote)
                quote = None
                continue
        else:
            opening = _v129_quote_at(answer, i)
            if opening:
                quote = opening
                i += len(opening)
                continue
            if char in "([{":
                stack.append(char)
            elif char in ")]}" and stack:
                stack.pop()
            elif not stack and char in ".;\n":
                # Decimal punctuation is part of the literal, not a sentence boundary.
                decimal = (
                    char == "."
                    and i + 1 < len(answer)
                    and answer[i + 1].isdigit()
                    and (i == 0 or answer[i - 1].isdigit() or answer[i - 1] in " +-")
                )
                if not decimal:
                    fragment = answer[start : i + 1].strip()
                    if fragment:
                        out.append(fragment)
                    start = i + 1
        i += 1
    tail = answer[start:].strip()
    if tail:
        out.append(tail)
    # A separated condition header still governs the following clause. Keep its
    # original contiguous text so a model cannot crop away the condition.
    merged = []
    pending_start = None
    cursor = 0
    for fragment in out:
        found = answer.find(fragment, cursor)
        end = found + len(fragment)
        cursor = end
        start = found if pending_start is None else pending_start
        combined = answer[start:end].strip()
        surface = _v129_without_quoted(combined)
        header = bool(
            re.match(r"\s*(?:if|unless|when|whenever|provided|assuming)\b", surface, re.I)
        )
        dangling = header and (
            combined.rstrip(".;\n ").endswith(":")
            or "," not in surface
            and not re.search(r"\b(?:returns?|outputs?|then)\b", surface, re.I)
        )
        if dangling:
            pending_start = start
        else:
            merged.append(combined)
            pending_start = None
    if pending_start is not None:
        merged.append(answer[pending_start:].strip())
    return merged


def _v129_without_quoted(text):
    # Predicate markers in string values, e.g. f('not'), are not language-level negations.
    chars = list(text)
    quote = None
    escape = False
    i = 0
    while i < len(text):
        if quote:
            if escape:
                escape = False
            elif text[i] == "\\":
                escape = True
            elif text.startswith(quote, i):
                for j in range(i, min(len(text), i + len(quote))):
                    chars[j] = " "
                i += len(quote)
                quote = None
                continue
            chars[i] = " "
        else:
            opening = _v129_quote_at(text, i)
            if opening:
                quote = opening
                for j in range(i, min(len(text), i + len(opening))):
                    chars[j] = " "
                i += len(opening)
                continue
        i += 1
    return "".join(chars)


def _v129_affirmation(answer):
    if not isinstance(answer, str) or re.search(r"[^a-zA-Z\s,.'!]", answer):
        return False
    normalized = re.sub(r"[^a-z ]", "", answer.lower()).strip()
    return normalized in {
        "yes",
        "correct",
        "that is correct",
        "yes that is correct",
        "yes thats correct",
        "yes correct",
        "yes exactly",
        "exactly",
        "yes it is",
        "yes that is the intended result",
    }


def _v129_expected(text):
    text = text.lstrip()
    if not text:
        raise ValueError("missing explicit output")
    if text[0] in "([{":
        end = _group_end(text, 0)
        if end is None:
            raise ValueError("incomplete literal")
    else:
        meaningful = [
            t
            for t in tokenize.generate_tokens(io.StringIO(text.splitlines()[0]).readline)
            if t.type not in {tokenize.NEWLINE, tokenize.NL, tokenize.ENDMARKER}
        ]
        if not meaningful:
            raise ValueError("missing output token")
        end = meaningful[0].end[1]
        if meaningful[0].string in {"+", "-"} and len(meaningful) > 1:
            end = meaningful[1].end[1]
    suffix = text[end:].lstrip()
    if re.match(r"[+*/%<>=!&|^~(\[]|-(?!>)|\.[A-Za-z_]|,\s*[-+0-9\[({\"\']", suffix):
        raise ValueError("computed or compound output is not an explicit literal")
    raw = text[:end]
    if re.fullmatch(r"[+-]?\d+\.", raw):
        raw = raw[:-1]
    return _read_literal(raw)


def _v129_explicit_relations(answer, entry):
    results = []
    for clause in _v129_clauses(answer):
        surface = _v129_without_quoted(clause)
        if re.search(
            r"\b(?:if|unless|except|otherwise|when|whenever|not|never|cannot|rather than|instead of|or|maybe|perhaps|approximately|provided|assuming|only)\b|n't\b",
            surface,
            re.I,
        ):
            continue
        for found in re.finditer(r"\b" + re.escape(entry) + r"\s*\(", clause):
            end = _group_end(clause, clause.index("(", found.start()))
            if end is None:
                continue
            try:
                call = ast.parse(clause[found.start() : end], mode="eval").body
                args = _call_literal(call, [entry])
                relation = re.match(
                    r"\s*(?:==|->|=>|should\s+return|must\s+return|returns?|outputs?)\s*",
                    clause[end:],
                    re.I,
                )
                if args is None or relation is None:
                    continue
                expected = _v129_expected(clause[end + relation.end() :])
                if not stated_in(results, args, expected):
                    results.append((args, expected))
            except (ValueError, TypeError, SyntaxError, RecursionError, tokenize.TokenError):
                continue
        for found in re.finditer(r"\bfor\s+(?:the\s+)?(input|arguments)\s*:?\s*", clause, re.I):
            tail = clause[found.end() :]
            for connector in re.finditer(
                r",?\s+(?:(?:it|the result|the output)\s+)?(?:should return|must return|returns?|should be|is|outputs?)\s+",
                tail,
                re.I,
            ):
                try:
                    value = _read_literal(tail[: connector.start()].strip().rstrip(",").strip())
                    args = (
                        tuple(value)
                        if found[1].lower() == "arguments" and isinstance(value, (tuple, list))
                        else (value,)
                    )
                    expected = _v129_expected(tail[connector.end() :])
                    if not stated_in(results, args, expected):
                        results.append((args, expected))
                    break
                except (ValueError, TypeError, SyntaxError, RecursionError, tokenize.TokenError):
                    continue
    return results


def _v129_artifact(source):
    module = ast.parse(source)
    kept = []
    for statement in module.body:
        demonstration = isinstance(statement, ast.Assert)
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            demonstration = (
                isinstance(statement.value.func, ast.Name) and statement.value.func.id == "print"
            )
        if isinstance(statement, ast.If):
            demonstration = "__name__" in {
                n.id for n in ast.walk(statement.test) if isinstance(n, ast.Name)
            } and any(
                isinstance(n, ast.Constant) and n.value == "__main__"
                for n in ast.walk(statement.test)
            )
        if not demonstration:
            kept.append(statement)
    module.body = kept
    return ast.unparse(module)


_V129_TYPES = {
    "bool": ("bool", "boolean"),
    "int": ("int", "integer"),
    "float": ("float",),
    "str": ("str", "string"),
    "list": ("list",),
    "tuple": ("tuple",),
    "dict": ("dict", "dictionary"),
    "set": ("set",),
}


def _v129_condition(value, span):
    if not isinstance(value, dict) or not isinstance(span, str) or not span.strip():
        return None
    kind = value.get("kind")
    index = value.get("index")
    if not isinstance(kind, str):
        return None
    if type(index) is not int or not 0 <= index < 8:
        return None
    lower = _v129_without_quoted(span).lower()
    if kind == "arg_empty" and type(value.get("empty")) is bool:
        if not re.search(r"\b(?:empty|nonempty|non-empty)\b", lower):
            return None
        negative = bool(re.search(r"\b(?:not\s+empty|nonempty|non-empty)\b", lower))
        if value["empty"] == negative:
            return None
        return dict(kind=kind, index=index, empty=value["empty"])
    if (
        kind == "arg_type"
        and isinstance(value.get("value"), str)
        and value.get("value") in _V129_TYPES
    ):
        if re.search(r"\b(?:not|never|unless)\b|n't\b", lower):
            return None
        if not any(re.search(r"\b" + word + r"\b", lower) for word in _V129_TYPES[value["value"]]):
            return None
        return dict(kind=kind, index=index, value=value["value"])
    if (
        kind == "arg_compare"
        and isinstance(value.get("op"), str)
        and value.get("op") in {"eq", "ne", "lt", "le", "gt", "ge"}
    ):
        try:
            raw = value.get("value")
            literal = _read_literal(raw) if isinstance(raw, str) else raw
            if type(literal) not in {type(None), bool, int, float, str}:
                return None
            if isinstance(literal, str) and len(literal) > 160:
                return None
            if isinstance(literal, (int, float)) and (
                not math.isfinite(literal) or abs(literal) > 1000000
            ):
                return None
        except (ValueError, TypeError, SyntaxError, RecursionError):
            return None
        operators = {
            "eq": r"==|\bequals?\b|\bequal to\b",
            "ne": r"!=|\bnot equal\b|\bdiffers? from\b",
            "lt": r"(?<![<>=])<(?![=>])|\bless than\b|\bbelow\b",
            "le": r"<=|\bat most\b|\bno more than\b",
            "gt": r"(?<![<>=])>(?![=>])|\bgreater than\b|\babove\b",
            "ge": r">=|\bat least\b|\bno less than\b",
        }
        # This checks a literal/operator spelling, not arbitrary natural-language entailment.
        if not re.search(operators[value["op"]], lower):
            return None
        if type(literal) in {int, float} and not re.search(
            r"(?<![\w.])" + re.escape(str(literal)) + r"(?![\w.])", span
        ):
            return None
        if type(literal) is str and literal not in span:
            return None
        return dict(kind=kind, index=index, op=value["op"], value=literal)
    return None


def quote(text, span):
    return isinstance(text, str) and isinstance(span, str) and bool(span.strip()) and span in text


def literal_relations(answer, entry):
    return _v129_explicit_relations(answer, entry)


def affirmation(answer):
    return _v129_affirmation(answer)


def artifact(source):
    return _v129_artifact(source)


def _v129_value_equal(a, b):
    if type(a) in (bool, int, float) and type(b) in (bool, int, float):
        try:
            if a == b:
                return True
            return (
                math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-6)
                if type(a) is float or type(b) is float
                else False
            )
        except (OverflowError, ValueError, TypeError):
            return False
    if type(a) is not type(b):
        return False
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(_v129_value_equal(x, y) for x, y in zip(a, b, strict=False))
    return a == b


def public_records(prompt, entry):
    records = []
    lines = prompt.splitlines(keepends=True)
    for args, expected in parse_examples(prompt, [entry])[:12]:
        span = ""
        for start in range(len(lines)):
            for width in (1, 2, 3, 4):
                fragment = "".join(lines[start : start + width])
                if stated_in(parse_examples(fragment, [entry]), args, expected):
                    span = fragment
                    break
            if span:
                break
        if span:
            records.append(
                dict(
                    id="P" + str(len(records)),
                    args=args,
                    expected=expected,
                    source_span=span,
                    source="prompt",
                    tier="P",
                    status="active",
                    authority="direct",
                    history=[],
                )
            )
    return records


def _v15010_property_sources(answer):
    """Bound a verbatim answer-clause prefix without cutting any clause."""
    clauses = _v129_clauses(answer)
    sources = []
    characters = 0
    for index, clause in enumerate(clauses):
        if len(sources) >= 32 or characters + len(clause) > 12000:
            break
        sources.append(dict(source_clause_index=index, source_span=clause))
        characters += len(clause)
    return dict(
        protocol="answer-property-clauses-v1",
        clauses=sources,
        total_clause_count=len(clauses),
        emitted_clause_count=len(sources),
        emitted_character_count=characters,
        omitted_clause_count=len(clauses) - len(sources),
        overflow=len(sources) < len(clauses),
        max_clauses=32,
        max_characters=12000,
    )


def _v15010_property_packet(packet):
    return dict(
        packet,
        answer_property_sources=_v15010_property_sources(packet.get("actual_answer", "")),
        property_node_contract=dict(
            required_fields=["kind", "value", "source_clause_index"],
            source_clause_index_type="integer, not bool; must occur in source table",
            kinds=["return_type"],
            return_type_values=["bool", "int", "float", "str", "list", "tuple", "dict", "set"],
            optional_condition=dict(
                arg_compare=dict(
                    fields=["kind", "index", "op", "value"],
                    op=["eq", "ne", "lt", "le", "gt", "ge"],
                    value="literal string",
                ),
                arg_empty=dict(fields=["kind", "index", "empty"], empty="boolean"),
                arg_type=dict(fields=["kind", "index", "value"], value_enum="return_type_values"),
                index="integer 0..7",
                condition_span="verbatim substring of cited clause",
                unconditional="omit condition/condition_span",
            ),
            forbidden_fields=[
                "source_span",
                "scope_clause",
                "expression",
                "expr",
                "code",
                "predicate",
            ],
            context_policy=(
                "Other clauses: independent input-type declarations or same-type output "
                "declarations only; otherwise unknown."
            ),
        ),
    )


def _v15010_type_contrast(clause, value):
    """Recognize one complete mandatory output-type assertion and its exclusion."""
    if not isinstance(value, str) or value not in _V129_TYPES:
        return False
    surface = _v129_without_quoted(clause).strip()
    names = "|".join(re.escape(word) for words in _V129_TYPES.values() for word in words)
    article = r"(?:(?:a|an|the|one|single|python|built-in|builtin)\s+)*"
    subject = (
        r"(?:(?:the\s+)?function\s+(?:must|should)\s+return\s+"
        r"|(?:the\s+)?(?:return(?:ed)?\s+type|output\s+type|result\s+type)\s+"
        r"(?:is|must be|should be)\s+)"
    )
    introduction = (
        r"(?:(?:Correction|To clarify)[:,]?\s*"
        r"|Your interpretation includes (?:some )?inaccuracies:\s*)?"
    )
    pattern = (
        introduction + subject + article + r"(?P<positive>" + names + r")\b"
        r"(?:\s*,\s*not\s+|\s+rather than\s+)" + article + r"(?P<negative>" + names + r")\b\s*[.;]?"
    )
    found = re.fullmatch(pattern, surface, re.I)
    if not found:
        return False
    positive = found["positive"].lower()
    negative = found["negative"].lower()
    canonical = {word: kind for kind, words in _V129_TYPES.items() for word in words}
    return canonical[positive] == value and canonical[positive] != canonical[negative]


def _v15010_indexed_type_subject(clause, value, conditional):
    """Admit a complete plain type declaration, not an embedded assertion."""
    surface = _v129_without_quoted(clause)
    if not conditional and re.search(
        r"\bfor\b[^.;\n]{0,100}\b(?:inputs?|arguments?|strings?|integers?|lists?|tuples?|dictionaries|sets?)\b",
        surface,
        re.I,
    ):
        return False
    names = (
        "|".join(re.escape(word) for word in _V129_TYPES.get(value, ()))
        if isinstance(value, str)
        else ""
    )
    if not names:
        return False
    article = r"(?:(?:a|an|the|one|single|python|built-in|builtin)\s+)*"
    subject = (
        r"(?:\b(?:the\s+)?function\s+(?:returns?|(?:must|should)\s+return)\s+"
        r"|\b(?:the\s+)?(?:output|result|return value|return(?:ed)? type|output type|result type)\s+"
        r"(?:is|must be|should be)\s+)"
    )
    introduction = r"(?:(?:Correction|To clarify)[:,]?\s*)?"
    # This optional tail only identifies collection members. A behavioral
    # explanation, reference to an old implementation or domain qualifier is
    # deliberately outside this small whole-clause grammar.
    members = (
        r"(?:\s+of\s+(?:integers?|strings?|floats?|booleans?|tuples?|lists?|dictionaries|sets?))?"
        if value in {"list", "tuple", "set"}
        else ""
    )
    return bool(
        re.fullmatch(
            r"\s*"
            + introduction
            + subject
            + article
            + r"(?:"
            + names
            + r")\b"
            + members
            + r"\s*[.;]?\s*",
            surface,
            re.I,
        )
    )


def _v15010_property_context(answer, clause, value):
    """Conservatively classify the rest of the answer; no new node is emitted."""
    if not isinstance(value, str) or value not in _V129_TYPES:
        return "Indexed return type value is outside the source protocol"
    types = r"(?:bool|boolean|int|integer|float|str|string|list|tuple|dict|dictionary|set)"
    article = r"(?:(?:a|an|one|single|any|python)\s+)*"
    members = (
        r"(?:\s+of\s+(?:integers?|strings?|floats?|booleans?|tuples?|lists?|dictionaries|sets?))?"
    )
    input_declaration = (
        r"\s*(?:(?:the\s+)?(?:input|argument)\s+(?:is|must be|should be)\s+"
        r"|(?:the\s+)?function\s+accepts\s+)" + article + types + members + r"\s*[.;]?\s*"
    )
    for other in _v129_clauses(answer):
        if other == clause:
            continue
        surface = _v129_without_quoted(other)
        if re.fullmatch(input_declaration, surface, re.I):
            continue
        if _v15010_type_contrast(other, value) or _v15010_indexed_type_subject(other, value, False):
            continue
        competing = any(
            _v15010_type_contrast(other, kind) or _v15010_indexed_type_subject(other, kind, False)
            for kind in _V129_TYPES
            if kind != value
        )
        if competing:
            return "Indexed property has conflicting_output_context"
        return "Indexed property has unclassified_context outside its governing clause"
    return None


def _v129_targeted_type(text, value):
    names = "|".join(re.escape(word) for word in _V129_TYPES[value])
    prefix = r"(?:(?:a|an|the|one|single|python|built-in|builtin)\s+)*"
    target = r"(?:\breturns?\s+|\boutputs?\s+|\b(?:return(?:ed)?\s+type|output\s+type|result\s+type)\s*(?:is|must be|should be|:)\s+|\b(?:result|output|return value)\s+(?:is|must be|should be)\s+)"
    pattern = target + prefix + r"(?P<type>" + names + r")\b"
    for found in re.finditer(pattern, text, re.I):
        tail = text[found.end() :]
        # A type word can modify an index/count instead of the result itself.
        if re.match(
            r"\s*(?:'s\b|index\b|indices\b|position\b|count\b|length\b|size\b|key\b)", tail, re.I
        ):
            continue
        return True
    return False


def _v129_property_clause(item, answer, prompt):
    span = item.get("source_span")
    for source, text in [("answer", answer), ("prompt", prompt)]:
        if not quote(text, span):
            continue
        clause = next((c for c in _v129_clauses(text) if span in c), None)
        if clause is None:
            continue
        requested_scope = item.get("scope_clause")
        if requested_scope is not None and (
            not isinstance(requested_scope, str)
            or span not in requested_scope
            or requested_scope not in clause
        ):
            return None
        return source, clause
    return None


def _v129_properties(items, answer, prompt, audit=None):
    accepted, advisory = [], []
    counts = dict(proposed=0, considered=0, known_index=0, invalid_index=0, legacy=0)
    source_table = _v15010_property_sources(answer)
    by_index = {
        item["source_clause_index"]: item["source_span"] for item in source_table["clauses"]
    }
    if not isinstance(items, list):
        if audit is not None:
            audit.update(counts, observed_list=False, accepted=0, rejection_counts={})
        return accepted, advisory
    counts["proposed"] = len(items)
    allowed = {
        "return_type",
        "length_equals_input",
        "members_from_input",
        "sorted_order",
        "unique_elements",
    }
    for index, item in enumerate(items[:12]):
        if not isinstance(item, dict):
            continue
        counts["considered"] += 1

        def uncertain(reason, item=item):
            advisory.append(
                dict(proposal=copy.deepcopy(item), reason=reason, source="model_interpretation")
            )

        indexed = "source_clause_index" in item
        if indexed:
            source_index = item.get("source_clause_index")
            if type(source_index) is not int or source_index not in by_index:
                counts["invalid_index"] += 1
                uncertain("Answer property source index is not an allowed integer index")
                continue
            counts["known_index"] += 1
        else:
            counts["legacy"] += 1
        if set(item) & {"expression", "expr", "code", "predicate"}:
            uncertain("Arbitrary model predicate is not admitted")
            continue
        if indexed:
            item = dict(
                item, source_span=by_index[source_index], scope_clause=by_index[source_index]
            )
        kind = item.get("kind")
        if indexed and kind != "return_type":
            uncertain("Indexed property kind is not supported by the source protocol")
            continue
        resolved = _v129_property_clause(item, answer, prompt)
        if not isinstance(kind, str) or kind not in allowed or resolved is None:
            uncertain("Property type or complete quoted scope is unavailable")
            continue
        source, clause = resolved
        condition = None
        condition_span = item.get("condition_span")
        if item.get("condition") is not None:
            if not quote(clause, condition_span):
                uncertain("Condition is not part of the complete source clause")
                continue
            condition = _v129_condition(item["condition"], condition_span)
            if condition is None:
                uncertain("Finite condition could not be checked against its source")
                continue
        surface = _v129_without_quoted(clause)
        if condition is not None:
            surface = surface.replace(_v129_without_quoted(condition_span), " ", 1)
        if kind == "unique_elements":
            surface = re.sub(
                r"\b(?:no duplicates?|without duplicates?)\b", "unique", surface, flags=re.I
            )
        contrast = (
            indexed
            and kind == "return_type"
            and condition is None
            and _v15010_type_contrast(clause, item.get("value"))
        )
        if indexed:
            context_rejection = _v15010_property_context(answer, clause, item.get("value"))
            if context_rejection:
                uncertain(context_rejection)
                continue
        if not contrast and re.search(
            r"\b(?:if|unless|except|otherwise|when|whenever|not|never|cannot|or|maybe|perhaps|approximately|provided|assuming)\b|n't\b",
            surface,
            re.I,
        ):
            uncertain("Conditional, alternative or negative scope remains unresolved")
            continue
        lower = (
            surface.lower()
            .replace("non-decreasing", "nondecreasing")
            .replace("non-increasing", "nonincreasing")
        )
        prop = dict(
            id="R" + str(index),
            kind=kind,
            source_span=item["source_span"],
            scope_clause=clause,
            source=source,
            tier="R" if source == "answer" else "P",
            authority="direct",
            status="active",
        )
        if indexed:
            prop["source_clause_index"] = source_index
        if condition is not None:
            prop.update(condition=condition, condition_span=condition_span)
        if kind == "return_type":
            value = item.get("value")
            if (
                indexed
                and not contrast
                and not _v15010_indexed_type_subject(surface, value, condition is not None)
            ):
                uncertain("Indexed return type lacks one unambiguous actual-output subject")
                continue
            if (
                not isinstance(value, str)
                or value not in _V129_TYPES
                or not _v129_targeted_type(lower, value)
            ):
                uncertain("The type word is not an unambiguous return-type object")
                continue
            prop["value"] = value
        elif kind == "sorted_order":
            value = item.get("value")
            directions = {
                "ascending": r"(?:ascending|increasing|nondecreasing)",
                "descending": r"(?:descending|decreasing|nonincreasing)",
            }
            if not isinstance(value, str) or value not in directions:
                uncertain("Unsupported order direction")
                continue
            direction = directions[value]
            direct = (
                r"\breturns?\s+(?:the\s+)?(?:elements|values|items|list|sequence|tuple)\s+(?:sorted\s+)?(?:in\s+)?"
                + direction
                + r"\b"
                r"|\b(?:result|output|returned (?:list|sequence|values))\s+(?:is|are|must be|should be)\s+(?:sorted\s+)?(?:in\s+)?"
                + direction
                + r"\b"
            )
            other = directions["descending" if value == "ascending" else "ascending"]
            if not re.search(direct, lower) or re.search(r"\b" + other + r"\b", lower):
                uncertain("Order does not clearly modify the returned sequence")
                continue
            prop["value"] = value
        elif kind == "unique_elements":
            direct = (
                r"\breturns?\s+(?:the\s+)?(?:unique|distinct)\s+(?:elements|values|items)\b"
                r"|\b(?:result|output|returned (?:list|sequence|values))\s+(?:is|are|must be|should be|contains?|must contain|should contain|has|must have)\s+(?:only\s+)?(?:unique|distinct)\b"
            )
            if not re.search(direct, lower):
                uncertain("Uniqueness does not clearly refer to output elements")
                continue
        else:
            arg_index = item.get("arg_index")
            if type(arg_index) is not int or not 0 <= arg_index < 8:
                uncertain("Input argument index is not bounded")
                continue
            target = bool(
                re.search(
                    r"\b(?:result|output|returned (?:list|sequence|values))\b|\breturn\s+(?:a\s+|the\s+)?(?:list|sequence|values|elements|items)\b",
                    lower,
                )
            )
            if not target or not re.search(r"\b(?:input|argument|source|original)\b", lower):
                uncertain("The input/output relation does not have clear subjects")
                continue
            if kind == "length_equals_input":
                direct = (
                    r"\b(?:same|equal)\s+(?:length|size)\s+(?:as|to)\s+(?:(?:the|an?|each)\s+)?(?:(?:original|given)\s+)?(?:input|argument)\b"
                    r"|\b(?:length|size)\s+(?:is\s+)?(?:equal\s+to|equals?|matches?)\s+(?:the\s+)?(?:original\s+)?(?:input|argument)(?:\'s)?\s+(?:length|size)\b"
                )
                if not re.search(direct, lower) or re.search(
                    r"[+*/%<>=-]|\b(?:twice|double|triple|times|sum|plus|minus|less|more|fewer|shorter|longer|excluding|padded)\b",
                    lower,
                ):
                    uncertain("Length equality is not directly stated")
                    continue
            elif not re.search(
                r"\b(?:elements?|items?|members?|values?)\b", lower
            ) or not re.search(r"\b(?:from|belong|drawn|only)\b", lower):
                uncertain("Input membership is not directly stated")
                continue
            prop["arg_index"] = arg_index
        key = {
            k: v
            for k, v in prop.items()
            if k
            not in {"id", "source", "source_span", "scope_clause", "source_clause_index", "tier"}
        }
        existing = next(
            (
                p
                for p in accepted
                if {
                    k: v
                    for k, v in p.items()
                    if k
                    not in {
                        "id",
                        "source",
                        "source_span",
                        "scope_clause",
                        "source_clause_index",
                        "tier",
                    }
                }
                == key
            ),
            None,
        )
        if existing is None:
            accepted.append(prop)
        elif existing["source"] == "prompt" and source == "answer":
            accepted[accepted.index(existing)] = prop
    # Actual reply source outranks old prompt claims of the same finite predicate.
    kept = []
    for prop in accepted:
        peers = [
            p
            for p in accepted
            if p["kind"] == prop["kind"]
            and p.get("arg_index") == prop.get("arg_index")
            and p.get("condition") == prop.get("condition")
        ]
        answer_peers = [p for p in peers if p["source"] == "answer"]
        if answer_peers and prop["source"] == "prompt":
            continue
        governing = answer_peers or peers
        if len({repr(p.get("value")) for p in governing}) > 1:
            advisory.append(
                dict(
                    proposal=prop, reason="Same-source property conflict", source="source_conflict"
                )
            )
            continue
        prop["id"] = "R" + str(len(kept))
        kept.append(prop)
    if audit is not None:
        rejection_counts = {}
        for item in advisory:
            reason = item["reason"]
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
        audit.update(
            counts,
            observed_list=True,
            unconsidered=max(0, len(items) - 12),
            accepted=len(kept),
            accepted_answer=sum(item["source"] == "answer" for item in kept),
            rejection_counts=rejection_counts,
            source_table=dict(
                emitted_clause_count=source_table["emitted_clause_count"],
                omitted_clause_count=source_table["omitted_clause_count"],
                overflow=source_table["overflow"],
            ),
        )
    return kept, advisory


def _v129_domain_evidence(answer, span):
    if not quote(answer, span):
        return False
    return bool(
        re.search(
            r"\b(?:inputs?|arguments?|parameters?|domain|accepts?|valid|invalid|allowed|permitted)\b",
            _v129_without_quoted(span),
            re.I,
        )
    )


def _v129_literal_type_conflict(prop, record, answer):
    # This is deliberately narrower than full source-consistency inference.
    if (
        prop.get("kind") != "return_type"
        or prop.get("condition") is not None
        or prop.get("source") != "answer"
        or prop.get("authority") != "direct"
        or record.get("tier") != "H"
        or record.get("status", "active") != "active"
        or record.get("source") != "answer"
        or record.get("authority") != "direct"
        or "expected" not in record
        or not quote(answer, record.get("source_span"))
    ):
        return False
    actual_type = type(record.get("expected"))
    declared = {
        "bool": bool,
        "int": int,
        "float": float,
        "str": str,
        "list": list,
        "tuple": tuple,
        "dict": dict,
        "set": set,
    }.get(prop.get("value"))
    known = (type(None), bool, int, float, str, bytes, list, tuple, dict, set)
    if declared is None or actual_type not in known or actual_type is declared:
        return False
    # A literal I/O value does not by itself impose an exact numeric Python type:
    # 1, 1.0 and True can match under the SDK's value comparison semantics.
    if actual_type in (bool, int, float) and declared in (bool, int, float):
        return False
    return True


def _compile_reply_base(payload, answer, prompt, entry, records, binding):
    payload = payload if isinstance(payload, dict) else {}
    records = copy.deepcopy(list(records))
    effect = payload.get("effect")
    if (
        not isinstance(effect, str)
        or effect not in {"confirm", "local_change", "redefine", "unclear"}
        or effect != "unclear"
        and not quote(answer, payload.get("effect_span"))
    ):
        effect = "unclear"
    hard = literal_relations(answer, entry)
    if (
        isinstance(binding, dict)
        and binding.get("atomic") is True
        and binding.get("claim_kind") == "exact_call"
        and isinstance(binding.get("args"), tuple)
        and "expected" in binding
        and affirmation(answer)
    ):
        if not stated_in(hard, binding["args"], binding["expected"]):
            hard.insert(0, (binding["args"], binding["expected"]))
    for args, expected in hard:
        if not any(
            r.get("tier") == "H"
            and _typed_equal(r["args"], args)
            and _v129_value_equal(r["expected"], expected)
            for r in records
        ):
            records.append(
                dict(
                    id="H" + str(len(records)),
                    args=args,
                    expected=expected,
                    source_span=answer,
                    source="answer",
                    tier="H",
                    status="active",
                    authority="direct",
                    history=[],
                )
            )
    # Model-proposed call_results provide no extra truth beyond literal source parsing.
    advisory = []
    for item in (
        payload.get("call_results", []) if isinstance(payload.get("call_results", []), list) else []
    ):
        if not isinstance(item, dict):
            continue
        try:
            args = _read_literal(item.get("argument_literals"))
            expected = _read_literal(item.get("expected_literal"))
            grounded = (
                isinstance(args, tuple)
                and stated_in(hard, args, expected)
                and quote(answer, item.get("source_span"))
            )
        except (TypeError, ValueError, SyntaxError, RecursionError):
            grounded = False
        if not grounded:
            advisory.append(
                dict(
                    proposal=copy.deepcopy(item),
                    reason="Exact call is not directly stated by the actual reply",
                    source="model_interpretation",
                )
            )
    actual = [r for r in records if r.get("tier") == "H" and r.get("status", "active") == "active"]
    conflicting = {
        r["id"]
        for r in actual
        if any(
            _typed_equal(r["args"], x["args"])
            and not _v129_value_equal(r["expected"], x["expected"])
            for x in actual
        )
    }
    for record in records:
        if record["id"] in conflicting:
            record.update(
                tier="U",
                status="unresolved",
                reason="Actual reply gives incompatible results for the same call",
            )
        elif record.get("tier") == "P" and any(
            x["id"] not in conflicting
            and _typed_equal(record["args"], x["args"])
            and not _v129_value_equal(record["expected"], x["expected"])
            for x in actual
        ):
            record.update(tier="X", status="superseded")
            record.setdefault("history", []).append(
                {"reason": "Direct actual-answer correction of this exact call"}
            )
    property_protocol = {}
    properties, uncertain = _v129_properties(
        payload.get("properties"), answer, prompt, audit=property_protocol
    )
    advisory.extend(uncertain)
    kept_properties = []
    for property_record in properties:
        conflicting_facts = [
            r["id"] for r in records if _v129_literal_type_conflict(property_record, r, answer)
        ]
        if conflicting_facts:
            advisory.append(
                dict(
                    proposal=property_record,
                    reason="Unconditional return type conflicts with an explicit actual-answer literal value",
                    source="source_conflict",
                    conflicting_io_ids=conflicting_facts,
                )
            )
        else:
            kept_properties.append(property_record)
    properties = kept_properties
    property_protocol["active_after_literal_conflicts"] = len(properties)
    property_protocol["literal_conflict_rejections"] = property_protocol.get("accepted", 0) - len(
        properties
    )
    disputes = []
    for item in (
        payload.get("example_dispositions", [])
        if isinstance(payload.get("example_dispositions", []), list)
        else []
    ):
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("status"), str)
            or item.get("status") not in {"superseded", "unresolved"}
        ):
            continue
        record = next(
            (
                r
                for r in records
                if r["id"] == item.get("id")
                and r.get("tier") == "P"
                and r.get("status", "active") == "active"
            ),
            None,
        )
        if (
            record
            and quote(answer, item.get("response_span"))
            and quote(record["source_span"], item.get("original_span"))
            and isinstance(item.get("covered_condition"), str)
            and item["covered_condition"].strip()
        ):
            disputes.append(copy.deepcopy(item))
    updates = []
    for item in (
        payload.get("input_updates", [])
        if isinstance(payload.get("input_updates", []), list)
        else []
    ):
        if (
            isinstance(item, dict)
            and isinstance(item.get("id"), str)
            and isinstance(item.get("action"), str)
            and item.get("action") in {"keep", "exclude", "uncertain"}
            and _v129_domain_evidence(answer, item.get("source_span"))
            and isinstance(item.get("reason"), str)
            and item["reason"].strip()
        ):
            updates.append(copy.deepcopy(item))
    return dict(
        effect=effect,
        records=records,
        properties=properties,
        advisory=advisory,
        disputes=disputes,
        input_updates=updates,
        property_protocol=property_protocol,
    )


def _v129_bounded_arguments(value):
    if not isinstance(value, tuple) or len(value) > 8 or len(repr(value)) > 1200:
        return False
    pending = [(value, 0)]
    count = 0
    while pending:
        item, depth = pending.pop()
        count += 1
        if count > 120 or depth > 7:
            return False
        if type(item) is dict:
            pending.extend((v, depth + 1) for pair in item.items() for v in pair)
        elif type(item) in (tuple, list, set):
            pending.extend((v, depth + 1) for v in item)
        elif type(item) in (str, bytes) and len(item) > 160:
            return False
        elif type(item) in (int, float, complex):
            try:
                if not math.isfinite(abs(item)) or abs(item) > 1000000:
                    return False
            except (ValueError, OverflowError):
                return False
        elif type(item) not in (type(None), bool, int, float, complex, str, bytes):
            return False
    return True


def parse_inputs(payload, packet, previous=(), updates=()):
    payload = payload if isinstance(payload, dict) else {}
    answer = packet.get("actual_answer", "")
    inputs = []
    used_ids = set()
    for previous_item in previous:
        if not isinstance(previous_item, dict) or not _v129_bounded_arguments(
            previous_item.get("args")
        ):
            continue
        item = copy.deepcopy(previous_item)
        if not isinstance(item.get("domain_status"), str) or item.get("domain_status") not in {
            "confirmed",
            "exploratory",
            "excluded",
        }:
            item["domain_status"] = "exploratory"
        if not isinstance(item.get("id"), str) or item["id"] in used_ids:
            number = len(inputs)
            while "I" + str(number) in used_ids:
                number += 1
            item["id"] = "I" + str(number)
        used_ids.add(item["id"])
        inputs.append(item)
    for update in updates:
        if (
            not isinstance(update, dict)
            or not _v129_domain_evidence(answer, update.get("source_span"))
            or not isinstance(update.get("reason"), str)
            or not update["reason"].strip()
        ):
            continue
        item = next((x for x in inputs if x["id"] == update.get("id")), None)
        mapping = {"keep": "confirmed", "exclude": "excluded", "uncertain": "exploratory"}
        if (
            item is not None
            and isinstance(update.get("action"), str)
            and update.get("action") in mapping
        ):
            item.update(
                domain_status=mapping[update["action"]], domain_update=copy.deepcopy(update)
            )
    for item in (
        payload.get("observations", [])[:12]
        if isinstance(payload.get("observations", []), list)
        else []
    ):
        if not isinstance(item, dict):
            continue
        try:
            args = _read_literal(item.get("argument_literals"))
        except (TypeError, ValueError, SyntaxError, RecursionError):
            continue
        if not _v129_bounded_arguments(args):
            continue
        source = item.get("source")
        text = (
            packet.get("original_prompt", "")
            if source == "prompt"
            else answer
            if source == "answer"
            else ""
        )
        grounded = quote(text, item.get("domain_evidence"))
        status = (
            "confirmed"
            if grounded
            and isinstance(item.get("domain_status"), str)
            and item.get("domain_status") in {"explicit", "confirmed"}
            else "exploratory"
        )
        duplicate = next((old for old in inputs if _typed_equal(old["args"], args)), None)
        if duplicate is not None:
            # An excluded call is not silently reintroduced by a new suggestion.
            reply_domain_override = bool(
                duplicate.get("domain_update") or duplicate.get("arbitration")
            )
            if (
                duplicate["domain_status"] != "excluded"
                and status == "confirmed"
                and duplicate["domain_status"] != "confirmed"
                and not reply_domain_override
            ):
                duplicate.update(
                    domain_status="confirmed",
                    source=source,
                    source_span=item.get("domain_evidence", ""),
                )
            continue
        index = len(inputs)
        while "I" + str(index) in used_ids:
            index += 1
        ident = "I" + str(index)
        used_ids.add(ident)
        inputs.append(
            dict(
                id=ident,
                args=args,
                source=source if grounded else "model",
                source_span=item.get("domain_evidence", "") if grounded else "",
                domain_status=status,
            )
        )
    # Bound the ledger, prioritizing usable confirmed inputs over speculative ones.
    order = {"confirmed": 0, "exploratory": 1, "excluded": 2}
    return sorted(inputs, key=lambda item: order[item["domain_status"]])[:12]


def apply_arbitration(decisions, answer, records, properties, inputs):
    records, properties, inputs = (
        copy.deepcopy(list(records)),
        copy.deepcopy(list(properties)),
        copy.deepcopy(list(inputs)),
    )
    decisions = decisions.get("decisions", []) if isinstance(decisions, dict) else decisions
    if not isinstance(decisions, list):
        return records, properties, inputs
    for decision in decisions[:24]:
        if not isinstance(decision, dict):
            continue
        # The current public protocol wins whenever a new field is present. A
        # contradictory legacy alias cannot silently undo that current decision.
        span = (
            decision.get("source_span")
            if "source_span" in decision
            else decision.get("response_span")
        )
        scope = decision.get("scope") if "scope" in decision else decision.get("covered_condition")
        reason = decision.get("reason") if "reason" in decision else decision.get("explanation")
        if not quote(answer, span) or not all(
            isinstance(value, str) and value.strip() for value in (scope, reason)
        ):
            continue
        if "action" in decision:
            proposed_action = decision.get("action")
            action = (
                {
                    "retain": "retain_active",
                    "supersede": "superseded_by_reply",
                    "uncertain": "unresolved",
                }.get(proposed_action)
                if isinstance(proposed_action, str)
                else None
            )
        else:
            action = decision.get("decision")
        if not isinstance(action, str):
            continue
        target = next((r for r in records + properties if r.get("id") == decision.get("id")), None)
        if target is not None:
            # Direct actual-answer facts do not lose authority through a later model vote.
            if target.get("source") != "prompt" or target.get("tier") not in {"P", "U"}:
                continue
            if not quote(target.get("source_span", ""), decision.get("original_span")):
                continue
            if action == "superseded_by_reply":
                target.update(tier="X", status="superseded")
            elif action == "unresolved":
                target.update(tier="U", status="unresolved")
            elif action == "retain_active":
                target.update(tier="P", status="active")
            else:
                continue
            target.setdefault("history", []).append(
                dict(
                    reason="Source-scoped arbitration; fallible semantic judgment",
                    decision=copy.deepcopy(decision),
                )
            )
        else:
            item = next((x for x in inputs if x.get("id") == decision.get("id")), None)
            mapping = {
                "exclude_input": "excluded",
                "uncertain_input": "exploratory",
                "retain_input": "confirmed",
                "superseded_by_reply": "excluded",
                "unresolved": "exploratory",
                "retain_active": "confirmed",
            }
            if item is not None and action in mapping and _v129_domain_evidence(answer, span):
                item.update(domain_status=mapping[action], arbitration=copy.deepcopy(decision))
    return records, properties, inputs


"""Project V129 contract protocol. Textual understanding and measured facts stay distinct."""


def _contract_shape(data, measured, prompt):
    if not isinstance(data, dict):
        return False
    for key in ("input_summary", "result_summary", "operation_summary"):
        value = data.get(key)
        if not isinstance(value, str) or not 3 <= len(value.strip()) <= 400 or "?" in value:
            return False
    selected = data.get("witness_id", "")
    if not isinstance(selected, str) or selected and selected not in {m["id"] for m in measured}:
        return False
    spans = data.get("conflict_spans", [])
    return (
        isinstance(spans, list)
        and len(spans) in (0, 2)
        and all(isinstance(s, str) and 3 <= len(s) <= 180 and quote(prompt, s) for s in spans)
        and (not spans or spans[0] != spans[1])
    )


def _v1508_summary(value):
    return isinstance(value, str) and 3 <= len(value.strip()) <= 400 and "?" not in value


def _v1508_contract_fields(data, eligible, prompt):
    """Keep independent usable fields; auxiliary metadata cannot erase summaries."""
    data = data if isinstance(data, dict) else {}
    fields = {
        key: data[key]
        for key in ("input_summary", "result_summary", "operation_summary")
        if _v1508_summary(data.get(key))
    }
    rejected = []
    witness = data.get("witness_id", "")
    if not isinstance(witness, str) or witness and witness not in {i["id"] for i in eligible}:
        rejected.append("witness_id")
        witness = ""
    fields["witness_id"] = witness
    spans = data.get("conflict_spans", [])
    valid = (
        isinstance(spans, list)
        and len(spans) in (0, 2)
        and all(isinstance(s, str) and 3 <= len(s) <= 180 and quote(prompt, s) for s in spans)
        and (not spans or spans[0] != spans[1])
    )
    fields["conflict_spans"] = spans if valid else []
    if not valid:
        rejected.append("conflict_spans")
    return fields, rejected


def _v1508_decisions(data, prompt):
    """Validate provenance and bounded structure, without claiming semantic entailment."""
    items = data.get("decisions", []) if isinstance(data, dict) else []
    accepted, rejected = [], []
    if not isinstance(items, list):
        return [], ["not_a_list"]
    seen = set()
    for item in items[:8]:
        if not isinstance(item, dict):
            rejected.append("not_an_object")
            continue
        if item.get("dimension") not in (
            "operation",
            "result",
            "input",
            "boundary",
            "conflict",
        ) or item.get("status") not in ("unresolved", "contradictory"):
            rejected.append("status_or_dimension")
            continue
        span = item.get("source_span")
        if not isinstance(span, str) or not 3 <= len(span) <= 240 or not quote(prompt, span):
            rejected.append("source_span")
            continue
        options = item.get("alternatives")
        if (
            not isinstance(options, list)
            or len(options) != 2
            or not all(
                isinstance(option, str) and 3 <= len(option.strip()) <= 160 and "\n" not in option
                for option in options
            )
        ):
            rejected.append("alternatives")
            continue
        if options[0].strip().casefold() == options[1].strip().casefold():
            rejected.append("identical_alternatives")
            continue
        if not all(
            isinstance(item.get(key), str)
            and 3 <= len(item[key].strip()) <= 180
            and "\n" not in item[key]
            for key in ("detail", "behavior_effect")
        ):
            rejected.append("detail_or_effect")
            continue
        key = (item["dimension"], item["detail"].strip().casefold())
        if key in seen:
            rejected.append("duplicate")
            continue
        seen.add(key)
        accepted.append(
            {
                key: item[key]
                for key in (
                    "dimension",
                    "status",
                    "source_span",
                    "detail",
                    "alternatives",
                    "behavior_effect",
                )
            }
        )
    # A supported text conflict takes priority over an incidental boundary question.
    order = {"conflict": 0, "operation": 1, "result": 2, "input": 3, "boundary": 4}
    accepted.sort(key=lambda item: order[item["dimension"]])
    if len(accepted) > 2:
        rejected.extend(["capacity"] * (len(accepted) - 2))
    if len(items) > 8:
        rejected.extend(["scan_capacity"] * (len(items) - 8))
    return accepted[:2], rejected


def _v1508_render_contract(data, entry, witness, source_sha, decisions, illustrative=False):
    if not decisions:
        return _render_contract(data, entry, witness, source_sha, illustrative)
    clauses = [
        dict(id="I0", kind="input", text=data["input_summary"].strip()),
        dict(id="O0", kind="result", text=data["result_summary"].strip()),
        dict(id="R0", kind="operation", text=data["operation_summary"].strip()),
    ]
    lines = [
        f"Please clarify these behavior decisions for {entry}.",
        "The alternatives are tentative, may both be wrong, and are not an exhaustive list.",
    ]
    for index, item in enumerate(decisions):
        label = f"D{index}"
        lines.extend(
            [
                f"{label} — {item['detail']}",
                "Relevant description: " + json.dumps(item["source_span"], ensure_ascii=False),
                "A: " + item["alternatives"][0] + "; B: " + item["alternatives"][1],
                "Behavior affected: " + item["behavior_effect"],
            ]
        )
        clauses.append(dict(id=label, kind="open_decision", text=item["detail"], **item))
    lines.extend(
        [
            "For context, the current draft has this fallible interpretation:",
            "I0 — Inputs: " + clauses[0]["text"],
            "O0 — Result representation: " + clauses[1]["text"],
            "R0 — Computation: " + clauses[2]["text"],
        ]
    )
    if data.get("conflict_spans"):
        lines.append(
            "Potentially incompatible passages: "
            + json.dumps(data["conflict_spans"], ensure_ascii=False)
        )
    lines.append(
        "For each D item, state the intended rule rather than only an option letter; "
        "if neither alternative fits, give the correct rule. Also correct any other part of the draft interpretation."
    )
    if illustrative:
        lines.append(
            "One small illustrative input and intended result may help specify the mapping; no hidden tests or reference program are requested."
        )
    question = "\n".join(lines)
    return question, dict(
        protocol="decision-contract-v1",
        atomic=False,
        claim_kind="contract",
        clauses=clauses,
        source_sha256=source_sha,
        question_sha256=hashlib.sha256(question.encode()).hexdigest(),
    )


def _contract_witnesses(anchor, report, inputs, measured):
    """Only an anchor's completed, domain-confirmed calls can be printed as examples."""
    if not report:
        return []
    digest = hashlib.sha256(anchor.encode()).hexdigest()
    indices = {
        i for i, row in enumerate(report["candidates"]) if row.get("v150_receipt_source") == digest
    }
    result = []
    for item in measured:
        if (
            "result" not in item
            or item.get("domain_status") != "confirmed"
            or not any(item["id"].startswith(f"d{i}:") for i in indices)
        ):
            continue
        if len(item["argument_literals"]) > 220 or len(item.get("result_literal", "")) > 180:
            continue
        peers = [
            other
            for other in measured
            if other.get("argument_literals") == item["argument_literals"]
            and other.get("domain_status") == "confirmed"
            and "result" in other
        ]
        entry = dict(item, separates_candidates=len({o["result_literal"] for o in peers}) > 1)
        result.append(entry)
    return sorted(result, key=lambda item: (not item["separates_candidates"], item["id"]))


def _fallback_contract(anchor, entry, witness):
    signature = "argument roles are not reliably summarized"
    try:
        node = next(
            n
            for n in ast.parse(anchor).body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == entry
        )
        signature = "declared parameters: " + ast.unparse(node.args)
    except (SyntaxError, StopIteration, TypeError):
        pass
    return dict(
        input_summary=signature,
        result_summary=(
            "the observed example has type " + v150_type_name(witness["result"])
            if witness
            else "the intended result representation remains unresolved"
        ),
        operation_summary="the central rule could not be reliably summarized from the draft",
        conflict_spans=[],
    )


def _render_contract(data, entry, witness, source_sha, illustrative_example=False):
    """Rendering owns the sample bytes; the model supplies descriptions, never measured values."""
    clauses = [
        dict(id="I0", kind="input", text=data["input_summary"].strip()),
        dict(id="O0", kind="result", text=data["result_summary"].strip()),
        dict(id="R0", kind="operation", text=data["operation_summary"].strip()),
    ]
    lines = [
        f"My current interpretation of {entry} is provisional:",
        "I0 — Inputs: " + clauses[0]["text"],
        "O0 — Result representation: " + clauses[1]["text"],
        "R0 — Computation: " + clauses[2]["text"],
    ]
    if witness:
        args = _read_literal(witness["argument_literals"])
        expected = witness["result"]
        rendered = call_text(entry, args) + " -> " + repr(expected)
        clauses.append(
            dict(
                id="E0",
                kind="measured_example",
                args=args,
                expected=expected,
                text=rendered,
                witness_id=witness["id"],
                source_sha256=source_sha,
                domain_status=witness["domain_status"],
            )
        )
        lines.append("E0 — Observed on that implementation, not yet a requirement: " + rendered)
    else:
        lines.append("No completed, domain-supported example is available; no output is asserted.")
    if data.get("conflict_spans"):
        lines.append(
            "The description also contains these potentially incompatible statements: "
            + json.dumps(data["conflict_spans"], ensure_ascii=False)
        )
    lines.append(
        "Which parts of this interpretation match your intended behavior, and what should be corrected in the others?"
    )
    if illustrative_example:
        lines[-1] += (
            " If a correction changes the mapping, one small illustrative input and intended result would help specify it, when practical; no hidden tests or reference program are requested."
        )
    question = "\n".join(lines)
    binding = dict(
        protocol="contract-links-v1",
        atomic=False,
        claim_kind="contract",
        clauses=clauses,
        question_sha256=hashlib.sha256(question.encode()).hexdigest(),
    )
    return question, binding


def _scoped_example_confirmation(span, answer, target):
    """A limited affirmative language recognizer, not a general entailment oracle."""
    clauses = _v129_clauses(answer)
    if span not in clauses:
        return False
    topic = r"\b(?:E0|example|sample|demonstration|observed\s+(?:call|result)|shown\s+(?:call|result))\b"
    unsafe = r"\b(?:not|no|never|incorrect|wrong|reject\w*|instead|except|unless|if|when|only|but|however|actually|might|maybe|perhaps|possibly|probably|unsure|uncertain)\b|n['’]t\b"
    # Do not cherry-pick an affirmative fragment from a later correction of the sample.
    for clause in clauses:
        surface = _v129_without_quoted(clause)
        if re.search(topic, surface, re.I) and re.search(unsafe, surface, re.I):
            return False
    surface = " ".join(_v129_without_quoted(span).strip().split()).rstrip(".!")
    subject = r"(?:E0|(?:the|that|this|your) (?:(?:shown|observed|displayed|worked) )?(?:example|sample|demonstration)|the (?:shown|observed|displayed) (?:call|result))"
    positive = rf"(?:yes,? )?{subject} (?:is|was) (?:correct|accurate|right|confirmed)"
    positive += rf"|I confirm (?:that )?{subject}(?: is (?:correct|accurate|right))?"
    positive += rf"|{subject} matches the intended result"
    return bool(re.fullmatch(positive, surface, re.I) and not re.search(unsafe, surface, re.I))


def compile_reply(payload, answer, prompt, entry, records, binding):
    """Literal facts win; scoped example acknowledgement can add exactly one hard check."""
    compiled = _compile_reply_base(payload, answer, prompt, entry, records, binding)
    payload = payload if isinstance(payload, dict) else {}
    binding = binding if isinstance(binding, dict) else {}
    clauses = {
        c["id"]: c
        for c in binding.get("clauses", [])
        if isinstance(c, dict) and isinstance(c.get("id"), str)
    }
    spans = _v129_clauses(answer)
    links = []
    for item in (
        payload.get("clause_responses", [])
        if isinstance(payload.get("clause_responses"), list)
        else []
    ):
        if not isinstance(item, dict) or not isinstance(item.get("clause_id"), str):
            continue
        target = clauses.get(item["clause_id"])
        if (
            not target
            or not isinstance(item.get("verdict"), str)
            or item["verdict"] not in {"confirmed", "corrected", "unresolved"}
            or not isinstance(item.get("source_span"), str)
            or item["source_span"] not in spans
        ):
            continue
        span = item["source_span"]
        subjects = {
            "I0": r"\b(?:I0|inputs?|arguments?|parameters?)\b",
            "O0": r"\b(?:O0|returns?|result|representation|format)\b",
            "R0": r"\b(?:R0|rule|computation|operation|mapping|algorithm)\b",
            "E0": r"\b(?:E0|example|sample|demonstration|shown|observed)\b",
            "D0": r"\bD0\b",
            "D1": r"\bD1\b",
        }
        scoped = bool(
            re.search(subjects.get(item["clause_id"], r"(?!)"), _v129_without_quoted(span), re.I)
        )
        if item["clause_id"] == "E0" and item["verdict"] == "confirmed":
            scoped = _scoped_example_confirmation(span, answer, target)
        links.append(
            dict(
                item,
                proposed_verdict=item["verdict"],
                verdict=item["verdict"] if scoped else "unresolved",
                status="semantic_only",
            )
        )
    sample = clauses.get("E0")
    # Source text can confirm the sole rendered example even if the model misses a link.
    if sample and not any(x["clause_id"] == "E0" for x in links):
        for span in spans:
            if _scoped_example_confirmation(span, answer, sample):
                links.append(
                    dict(
                        clause_id="E0",
                        verdict="confirmed",
                        source_span=span,
                        status="semantic_only",
                    )
                )
                break
    sample_links = [x for x in links if x["clause_id"] == "E0"]
    if (
        sample
        and binding.get("protocol") == "contract-links-v1"
        and sample.get("kind") == "measured_example"
        and sample.get("domain_status") == "confirmed"
        and isinstance(sample.get("args"), tuple)
        and "expected" in sample
        and sample_links
        and all(x["verdict"] == "confirmed" for x in sample_links)
    ):
        explicit = literal_relations(answer, entry)
        conflicting = any(
            _typed_equal(args, sample["args"]) and not _v129_value_equal(value, sample["expected"])
            for args, value in explicit
        )
        supported = [
            x
            for x in sample_links
            if _scoped_example_confirmation(x["source_span"], answer, sample)
        ]
        if supported:
            proposed = dict(
                tier="H",
                status="active",
                source="answer",
                authority="direct",
                expected=sample["expected"],
                source_span=supported[0]["source_span"],
            )
            if any(
                _v129_literal_type_conflict(prop, proposed, answer)
                for prop in compiled["properties"]
            ):
                for link in supported:
                    link.update(
                        verdict="unresolved",
                        reason="Example acknowledgement conflicts with a scoped return-type requirement",
                    )
                supported = []
        if supported and not conflicting:
            already = any(
                r.get("tier") == "H"
                and _typed_equal(r["args"], sample["args"])
                and _v129_value_equal(r["expected"], sample["expected"])
                for r in compiled["records"]
            )
            if not already:
                compiled["records"].append(
                    dict(
                        id="H-confirmed-E0",
                        args=sample["args"],
                        expected=sample["expected"],
                        source="answer",
                        tier="H",
                        status="active",
                        authority="direct",
                        source_span=supported[0]["source_span"],
                        history=[],
                        question_link=dict(
                            clause_id="E0",
                            question_sha256=binding["question_sha256"],
                            witness_id=sample["witness_id"],
                            source_sha256=sample["source_sha256"],
                        ),
                    )
                )
            for link in supported:
                link["status"] = "compiled_exact_example"
            for record in compiled["records"]:
                if (
                    record.get("tier") == "P"
                    and _typed_equal(record["args"], sample["args"])
                    and not _v129_value_equal(record["expected"], sample["expected"])
                ):
                    record.update(tier="X", status="superseded")
                    record.setdefault("history", []).append(
                        dict(reason="Scoped confirmation of measured E0")
                    )
    notes = []
    for note in (
        payload.get("semantic_notes", []) if isinstance(payload.get("semantic_notes"), list) else []
    ):
        if (
            isinstance(note, dict)
            and isinstance(note.get("source_span"), str)
            and note["source_span"] in spans
            and isinstance(note.get("interpretation"), str)
            and note["interpretation"].strip()
        ):
            notes.append(
                dict(
                    source_span=note["source_span"],
                    interpretation=note["interpretation"],
                    authority="advisory",
                )
            )
    compiled["reply_coverage"] = dict(
        clause_responses=links,
        retained_reply_clauses=spans,
        semantic_notes=notes,
        decision_questions=[c for c in clauses.values() if c.get("kind") == "open_decision"],
        unresolved_clause_ids=[
            cid
            for cid in clauses
            if not any(x["clause_id"] == cid and x["verdict"] != "unresolved" for x in links)
        ],
        note="Semantic links are fallible interpretations; only compiled checks are executable. The full reply remains authoritative.",
    )
    return compiled


def _v129_digest(value):
    return hashlib.sha256(repr(value).encode()).hexdigest()


def _v129_source_sha(source):
    return hashlib.sha256(source.encode()).hexdigest()


def v150_type_name(value):
    """Return the ordinary type label through the public stdlib documentation API.

    Only literal-result and caught-exception types reach this helper. It does
    not inspect SDK objects, hidden requirements, or private SDK state.
    """
    return pydoc.describe(type(value)).removeprefix("class ")


def v150_prompt_packet(packet):
    """Keep V150's model-visible receipt names after internal field renaming.

    Only top-level metadata on measured-execution rows is translated. Literal
    result values, nested user dictionaries, source strings and all other
    packet fields are left alone. Insertion order matches the original JSON.
    These legacy names describe our own receipts, not private SDK fields.
    """
    measurement = packet.get("measured_execution")
    if not isinstance(measurement, dict):
        return packet
    names = {"v150_receipt_context": "_context_key", "v150_receipt_source": "_source_sha"}

    def original_row(row):
        if not isinstance(row, dict):
            return row
        return {names.get(key, key): value for key, value in row.items()}

    if isinstance(measurement.get("candidates"), list):
        measurement = dict(
            measurement, candidates=[original_row(row) for row in measurement["candidates"]]
        )
    else:
        measurement = original_row(measurement)
    return dict(packet, measured_execution=measurement)


class _V129Session:
    CODE_ROLES = {
        "initial_builder",
        "preserving_builder",
        "reconstruction_builder",
        "boundary_builder",
        "obligation_rescue",
        "execution_patch",
    }

    def __init__(self, algorithm, env):
        self.env = env
        self.policy = algorithm.config
        self.cap = min(17, max(1, int(algorithm.config["max_llm_calls"])))
        self.exec_cap = min(7, max(1, int(algorithm.config["max_exec_calls"])))
        self.calls = self.executions = 0
        self.recovered = self.arbitrated = self.completed_unknown = False
        self.lock = threading.Lock()
        self.packet = {}
        self.cache = {}
        self.events = []
        self.binding = {}

    def remaining(self):
        return self.cap - self.calls

    @staticmethod
    def decode(text):
        if not isinstance(text, str):
            return None
        text = text.strip()
        if text.startswith("```") and text.endswith("```"):
            text = "\n".join(text.splitlines()[1:-1]).strip()
        try:
            value = json.loads(text)
            return value if isinstance(value, dict) else None
        except (ValueError, RecursionError):
            return None

    @staticmethod
    def schema_ok(role, data):
        if not isinstance(data, dict):
            return False
        if role in {"behavior_question", "question_revision"}:
            # Partial records reach the field-level question validator for repair.
            return any(
                isinstance(data.get(k), str)
                for k in ("input_summary", "result_summary", "operation_summary")
            ) or isinstance(data.get("decisions"), list)
        if role == "candidate_review":
            return (
                type(data.get("candidate_index")) is int
                and isinstance(data.get("source_span"), str)
                and isinstance(data.get("reason"), str)
            )
        if role == "implementation_plan":
            return isinstance(data.get("routes"), list)
        if role == "observation_inputs":
            return isinstance(data.get("observations"), list)
        if role == "reply_map":
            return all(isinstance(data.get(k), list) for k in ("call_results", "properties"))
        if role == "evidence_arbitration":
            return isinstance(data.get("decisions"), list)
        if role == "reply_compliance_review":
            return (
                isinstance(data.get("decision"), str)
                and data["decision"] in {"retain", "propose"}
                and (
                    data["decision"] == "retain"
                    or all(
                        isinstance(data.get(k), str)
                        for k in ("source_span", "replacement_source", "reason")
                    )
                )
            )
        return False

    def request(self, role, packet, reserve=0, recover=True):
        with self.lock:
            if self.calls >= self.cap - reserve or self.env.prompt_cost >= self.env.prompt_budget:
                self.events.append(dict(role=role, status="budget_abstention"))
                return {}
            self.calls += 1
        if role == "reply_map":
            packet = _v15010_property_packet(packet)
            table = packet["answer_property_sources"]
            self.events.append(
                dict(
                    role="v15010_property_sources",
                    status="measured",
                    protocol=table["protocol"],
                    emitted_clause_count=table["emitted_clause_count"],
                    emitted_character_count=table["emitted_character_count"],
                    omitted_clause_count=table["omitted_clause_count"],
                    overflow=table["overflow"],
                )
            )
        packet = _campaign_packet(role, packet, self.policy)
        target_schema = (
            packet.get("expected_schema", {})
            if role == "schema_recovery"
            else V129_PROMPTS.get("schemas", {}).get(role, {})
        )
        packet = dict(packet, expected_schema=target_schema)
        messages = [
            dict(role="system", content=V129_PROMPTS["roles"][role]),
            dict(
                role="user",
                content=V129_PROMPTS["user_envelope"].replace(
                    "{packet_json}",
                    json.dumps(v150_prompt_packet(packet), ensure_ascii=False, default=repr),
                ),
            ),
        ]
        try:
            text = self.env.llm(messages)
        except Exception as exc:
            self.events.append(
                dict(role=role, status="llm_unavailable", exception=v150_type_name(exc))
            )
            return {}
        if role == "schema_recovery":
            return {"raw_payload": text} if isinstance(text, str) else {}
        if role in self.CODE_ROLES:
            code = extract_code(text, self.packet["entry_point"])
            if code:
                return {"python_source": artifact(code)}
        else:
            obj = self.decode(text)
            if self.schema_ok(role, obj):
                return obj
        self.events.append(dict(role=role, status="invalid_representation"))
        if recover and isinstance(text, str) and text:
            with self.lock:
                allowed = not self.recovered
                if allowed:
                    self.recovered = True
            if allowed:
                fixed = self.request(
                    "schema_recovery",
                    dict(
                        packet if role == "reply_map" else {},
                        previous_payload=text,
                        target_role=role,
                        output_kind="python_code_block"
                        if role in self.CODE_ROLES
                        else "json_object",
                        expected_schema=V129_PROMPTS.get("schemas", {}).get(role),
                        entry_point=self.packet["entry_point"],
                    ),
                    reserve=reserve,
                    recover=False,
                )
                raw = fixed.get("raw_payload", "")
                if role in self.CODE_ROLES:
                    code = extract_code(raw, self.packet["entry_point"])
                    if code:
                        return {"python_source": artifact(code)}
                else:
                    obj = self.decode(raw)
                    if self.schema_ok(role, obj):
                        return obj
        return {}

    def batch(self, tasks):
        if not tasks:
            return []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(tasks))) as pool:
            return list(pool.map(lambda task: self.request(*task), tasks))

    @staticmethod
    def code(payload):
        return payload.get("python_source") if isinstance(payload, dict) else None

    @staticmethod
    def v150_complete(row):
        failure = row.get("load_error")
        if failure:
            return not failure.get("skipped")
        return all(
            type(e.get("ok")) is bool and not (e.get("error") or {}).get("skipped")
            for field in ("checks", "observations")
            for e in row.get(field, [])
        ) and all(
            p.get("unknown_count") == 0
            and (
                type(p.get("ok")) is bool
                or (p.get("error") or {}).get("type") == "NoApplicableInput"
            )
            for p in row.get("properties", [])
        )

    @staticmethod
    def v150_domain_properties(row, inputs, properties):
        # Exploratory calls can be observed but cannot make a specification check fail.
        aggregates = []
        for prop in properties:
            cases = []
            for item, observation in zip(inputs, row["observations"], strict=False):
                if item.get("domain_status") != "confirmed":
                    continue
                events = observation.get("properties", [])
                found = (
                    [e for e in events if isinstance(e, dict) and e.get("id") == prop["id"]]
                    if isinstance(events, list)
                    else []
                )
                if len(found) != 1:
                    cases.append(
                        dict(
                            ok=None, applicable=None, error=dict(type="MissingPropertyObservation")
                        )
                    )
                else:
                    cases.append(found[0])
            applicable = [e for e in cases if e.get("applicable") is not False]
            covered = sum(type(e.get("ok")) is bool for e in applicable)
            failed = any(e.get("ok") is False for e in applicable)
            unknown_count = sum(type(e.get("ok")) is not bool for e in applicable)
            unknown = unknown_count > 0
            value = False if failed else None if unknown or not covered else True
            error = next((e.get("error") for e in applicable if e.get("error")), None)
            if not applicable:
                error = dict(
                    type="NoApplicableInput", message="No confirmed applicable input was supplied"
                )
            # A counterexample settles truth, not coverage of the other cases.
            aggregates.append(
                dict(
                    id=prop["id"],
                    ok=value,
                    covered=covered,
                    error=error,
                    unknown_count=unknown_count,
                )
            )
        row["properties"] = aggregates

    @staticmethod
    def unknown_row(records, inputs, reason):
        return dict(
            load_error=dict(type="Unknown", message=reason, skipped=True),
            checks=[dict(ok=None, error=dict(skipped=True)) for _ in records],
            observations=[dict(ok=None, error=dict(skipped=True)) for _ in inputs],
            properties=[],
        )

    def execute(self, codes, records, inputs, properties=(), force=False):
        if not codes:
            return None
        inputs = [i for i in inputs if i.get("domain_status") != "excluded"]
        context = ClariBridge.v150_context_key(records, inputs, properties)
        hashes = [_v129_source_sha(c) for c in codes]
        rows = [copy.deepcopy(self.cache.get((h, context))) if not force else None for h in hashes]
        missing = [i for i, r in enumerate(rows) if r is None]
        if missing:
            with self.lock:
                allowed = self.executions < self.exec_cap
                if allowed:
                    self.executions += 1
            if allowed:
                request = dict(
                    entry=self.packet["entry_point"],
                    candidates=[codes[i] for i in missing],
                    checks=[(r["args"], r["expected"]) for r in records],
                    inputs=[i["args"] for i in inputs],
                    properties=list(properties),
                    per_call_seconds=2,
                    total_seconds=25,
                )
                try:
                    output = (
                        self.env.exec_code(
                            "EXECUTION_REQUEST = "
                            + repr(request)
                            + "\n"
                            + INDEPENDENT_EXECUTION_PROGRAM
                        )
                        or ""
                    )
                    blobs = [
                        line.split("@@EVIDENCE_LEDGER@@", 1)[1]
                        for line in output.splitlines()
                        if line.startswith("@@EVIDENCE_LEDGER@@")
                    ]
                    data = json.loads(blobs[-1])
                    measured = data["candidates"]
                    if len(measured) != len(missing) or data.get("configuration_error"):
                        raise ValueError("Execution response cannot be aligned")
                    for index, row in zip(missing, measured, strict=False):
                        if not isinstance(row, dict) or any(
                            not isinstance(row.get(field), list) or len(row[field]) != size
                            for field, size in [
                                ("checks", len(records)),
                                ("observations", len(inputs)),
                            ]
                        ):
                            continue
                        if row.get("load_error") is not None and not isinstance(
                            row["load_error"], dict
                        ):
                            continue
                        for field in ("checks", "observations"):
                            for k, event in enumerate(row[field]):
                                if (
                                    not isinstance(event, dict)
                                    or event.get("ok") is not None
                                    and type(event.get("ok")) is not bool
                                ):
                                    row[field][k] = dict(
                                        ok=None,
                                        error=dict(skipped=True, type="MalformedMeasurement"),
                                    )
                                elif str((event.get("error") or {}).get("message", "")).startswith(
                                    "Unsupported result type:"
                                ):
                                    event["ok"] = None
                                    event["error"]["skipped"] = True
                                    event["error"]["stage"] = "result_representation"
                        if not isinstance(row.get("properties"), list):
                            row["properties"] = []
                        self.v150_domain_properties(row, inputs, properties)
                        rows[index] = row
                except Exception as exc:
                    self.events.append(
                        dict(role="execution", status="tool_unknown", exception=v150_type_name(exc))
                    )
            else:
                self.events.append(dict(role="execution", status="budget_unknown"))
        for index, row in enumerate(rows):
            if row is None:
                row = self.unknown_row(records, inputs, "No aligned measurement available")
                rows[index] = row
            row["v150_receipt_context"] = context
            row["v150_receipt_source"] = hashes[index]
            if self.v150_complete(row):
                self.cache[(hashes[index], context)] = copy.deepcopy(row)
        return dict(
            candidates=rows,
            context_key=context,
            source_sha_by_index=hashes,
            inputs=copy.deepcopy(inputs),
        )


def _v132_length_facade(source, entry):
    """Add one narrow public-call fallback without changing the original body.

    The original callable is always used for arguments its signature accepts.
    A failed binding may drop one redundant trailing collection length. Function
    body exceptions are not intercepted. Self-referential or ambiguous entries
    are left alone so that recursion and module aliases keep their prior meaning.
    """
    try:
        module = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return source
    definitions = [
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == entry
    ]
    if len(definitions) != 1 or definitions[0].decorator_list:
        return source
    if any(isinstance(node, ast.Name) and node.id == entry for node in ast.walk(module)):
        return source
    arguments = definitions[0].args
    if arguments.vararg or arguments.kwarg or arguments.kwonlyargs:
        return source
    if not arguments.posonlyargs and not arguments.args:
        return source
    stem = "_v132_" + hashlib.sha256(source.encode()).hexdigest()[:16]
    if stem in source:
        return source
    extra = f"""
import builtins as {stem}_builtins
import inspect as {stem}_inspect
{stem}_target = {entry}
try:
    {stem}_signature = {stem}_inspect.signature({stem}_target)
except (TypeError, ValueError):
    {stem}_signature = None

def {entry}(*_v132_values, **_v132_keywords):
    if {stem}_signature is None:
        return {stem}_target(*_v132_values, **_v132_keywords)
    try:
        {stem}_signature.bind(*_v132_values, **_v132_keywords)
    except TypeError:
        _v132_parameters = tuple({stem}_signature.parameters.values())
        _v132_capacity = sum(_p.kind in (_p.POSITIONAL_ONLY, _p.POSITIONAL_OR_KEYWORD)
                             for _p in _v132_parameters)
        _v132_simple = not any(_p.kind == _p.VAR_POSITIONAL for _p in _v132_parameters)
        _v132_companion = (
            not _v132_keywords and _v132_simple
            and {stem}_builtins.len(_v132_values) == _v132_capacity + 1
            and {stem}_builtins.type(_v132_values[-1]) is {stem}_builtins.int
            and {stem}_builtins.type(_v132_values[-2]) in
                ({stem}_builtins.list, {stem}_builtins.tuple, {stem}_builtins.str)
            and _v132_values[-1] == {stem}_builtins.len(_v132_values[-2])
        )
        if not _v132_companion:
            raise
        {stem}_signature.bind(*_v132_values[:-1])
        return {stem}_target(*_v132_values[:-1])
    return {stem}_target(*_v132_values, **_v132_keywords)
"""
    return source.rstrip() + "\n" + extra


class ClariBridge(ClarificationAlgorithmBase):
    DEFAULT_CONFIG = {
        "max_llm_calls": 17,
        "max_exec_calls": 7,
        "raw_reconstruction": False,
        "illustrative_example": True,
        "literal_clauses": False,
        "semantic_alternative": False,
        "no_consensus": False,
        "reviewers": 0,
        "disagreement_rescue": False,
        "fresh_candidates": 3,
        "planner": False,
        "arbiter": False,
        "branch_repair": False,
    }

    @staticmethod
    def v150_context_key(records, inputs, properties=()):
        # Includes domains and authority, not only values; a changed spec invalidates receipts.
        return _v129_digest(
            (records, [i for i in inputs if i.get("domain_status") != "excluded"], list(properties))
        )

    @staticmethod
    def v150_active(records):
        return [
            r
            for r in records
            if r.get("tier") in {"H", "P"} and r.get("status", "active") == "active"
        ]

    @staticmethod
    def v150_loaded(row):
        return row.get("load_error") is None

    @staticmethod
    def v150_property_values(row, properties):
        found = {}
        for event in row.get("properties", []):
            if not isinstance(event, dict):
                continue
            value = event.get("ok")
            if (
                type(event.get("covered")) is not int
                or event["covered"] < 1
                or type(value) is not bool
            ):
                value = None
            ident = event.get("id")
            found[ident] = value if ident not in found or found[ident] is value else None
        return [found.get(p.get("id")) for p in properties]

    @staticmethod
    def v150_usable_error(event):
        error = event.get("error") or {}
        return event.get("ok") is False and not error.get("skipped")

    def v150_score(self, row, records, properties=(), inputs=()):
        if not self.v150_loaded(row):
            return (-1,) * 15
        hard = [row["checks"][i].get("ok") for i, r in enumerate(records) if r.get("tier") == "H"]
        soft = [row["checks"][i].get("ok") for i, r in enumerate(records) if r.get("tier") != "H"]
        pv = self.v150_property_values(row, properties)
        # Both reply groups outrank old examples; interpreted properties cannot
        # outvote a directly written I/O fact by sheer number of predicates.
        reply_rules = [
            v
            for p, v in zip(properties, pv, strict=False)
            if p.get("source") == "answer" and p.get("authority", "direct") == "direct"
        ]
        soft += [
            v
            for p, v in zip(properties, pv, strict=False)
            if p.get("source") != "answer" and p.get("authority", "direct") == "direct"
        ]
        valid = [
            e.get("ok")
            for i, e in enumerate(row.get("observations", []))
            if i < len(inputs) and inputs[i].get("domain_status") == "confirmed"
        ]
        hp, hf, hc = (
            sum(v is True for v in hard),
            sum(v is False for v in hard),
            sum(type(v) is bool for v in hard),
        )
        rp, rf, rc = (
            sum(v is True for v in reply_rules),
            sum(v is False for v in reply_rules),
            sum(type(v) is bool for v in reply_rules),
        )
        sp, sf, sc = (
            sum(v is True for v in soft),
            sum(v is False for v in soft),
            sum(type(v) is bool for v in soft),
        )
        return (
            1,
            int(hp == len(hard)),
            int(hc == len(hard)),
            hp,
            -hf,
            hc,
            int(rp == len(reply_rules)),
            int(rc == len(reply_rules)),
            rp,
            -rf,
            rc,
            sp,
            -sf,
            sc,
            -sum(v is False for v in valid),
            sum(v is True for v in valid),
            sum(type(v) is bool for v in valid),
        )

    def v150_support(self, report, index, inputs):
        own = report["candidates"][index]
        if not self.v150_loaded(own):
            return 0
        keys = [
            (i, e.get("result_key"))
            for i, e in enumerate(own.get("observations", []))
            if i < len(inputs)
            and inputs[i].get("domain_status") == "confirmed"
            and e.get("ok") is True
            and e.get("result_key")
        ]
        if not keys:
            return 0
        seen = set()
        for row in report["candidates"]:
            if self.v150_loaded(row) and all(
                i < len(row["observations"])
                and row["observations"][i].get("ok") is True
                and row["observations"][i].get("result_key") == value
                for i, value in keys
            ):
                seen.add(row.get("v150_receipt_source"))
        return len(seen)

    def v150_rank(self, report, records, properties=(), inputs=()):
        inputs = [i for i in inputs if i.get("domain_status") != "excluded"]
        return max(
            range(len(report["candidates"])),
            key=lambda i: (
                self.v150_score(report["candidates"][i], records, properties, inputs),
                self.v150_support(report, i, inputs),
                -i,
            ),
        )

    @staticmethod
    def v150_grounded_failure(row, records, properties):
        if not row:
            return False
        load_error = row.get("load_error") or {}
        if load_error and not load_error.get("skipped"):
            return True
        for record, event in zip(records, row.get("checks", []), strict=False):
            if (
                record.get("authority", "direct") == "direct"
                and event.get("ok") is False
                and not (event.get("error") or {}).get("skipped")
            ):
                return True
        for property_record, value in zip(
            properties,
            ClariBridge.v150_property_values(row, properties),
            strict=False,
        ):
            if property_record.get("authority", "direct") == "direct" and value is False:
                return True
        return False

    @staticmethod
    def v150_grounded_gain(old, new, records, properties):
        load_error = old.get("load_error") or {}
        if load_error:
            return not load_error.get("skipped") and new.get("load_error") is None
        for record, before, after in zip(
            records, old.get("checks", []), new.get("checks", []), strict=False
        ):
            if (
                record.get("authority", "direct") == "direct"
                and before.get("ok") is False
                and not (before.get("error") or {}).get("skipped")
                and after.get("ok") is True
            ):
                return True
        old_values = ClariBridge.v150_property_values(old, properties)
        new_values = ClariBridge.v150_property_values(new, properties)
        return any(
            prop.get("authority", "direct") == "direct" and a is False and b is True
            for prop, a, b in zip(properties, old_values, new_values, strict=False)
        )

    def v150_can_accept(self, old, new, records, properties=(), inputs=(), semantic_evidence=False):
        if old.get("v150_receipt_context") != new.get(
            "v150_receipt_context"
        ) or not self.v150_loaded(new):
            return False
        if not self.v150_loaded(old):
            return self.v150_grounded_gain(old, new, records, properties)
        for previous, current in zip(old["checks"], new["checks"], strict=False):
            if type(previous.get("ok")) is bool and type(current.get("ok")) is not bool:
                return False
        for i, r in enumerate(records):
            if (
                r.get("tier") == "H"
                and old["checks"][i].get("ok") is True
                and new["checks"][i].get("ok") is not True
            ):
                return False
        op, np = (
            self.v150_property_values(old, properties),
            self.v150_property_values(new, properties),
        )
        for p, a, b in zip(properties, op, np, strict=False):
            if type(a) is bool and type(b) is not bool:
                return False
            if p.get("source") == "answer" and a is True and b is not True:
                return False
            old_covered = next(
                (e.get("covered", 0) for e in old.get("properties", []) if e.get("id") == p["id"]),
                0,
            )
            new_covered = next(
                (e.get("covered", 0) for e in new.get("properties", []) if e.get("id") == p["id"]),
                0,
            )
            if (
                type(old_covered) is int
                and old_covered > 0
                and (type(new_covered) is not int or new_covered < old_covered)
            ):
                return False
        live = [i for i in inputs if i.get("domain_status") != "excluded"]
        for i, item in enumerate(live):
            if item.get("domain_status") != "confirmed":
                continue
            a, b = old["observations"][i].get("ok"), new["observations"][i].get("ok")
            if type(a) is bool and type(b) is not bool or a is True and b is not True:
                return False
            for p in properties:
                before_prop = next(
                    (
                        e.get("ok")
                        for e in old["observations"][i].get("properties", [])
                        if e.get("id") == p["id"]
                    ),
                    None,
                )
                after_prop = next(
                    (
                        e.get("ok")
                        for e in new["observations"][i].get("properties", [])
                        if e.get("id") == p["id"]
                    ),
                    None,
                )
                if type(before_prop) is bool and type(after_prop) is not bool:
                    return False
        before, after = (
            self.v150_score(old, records, properties, live),
            self.v150_score(new, records, properties, live),
        )
        # A model citation or a formerly crashing probe is not an output oracle.
        # Accept a source edit only when a direct, executable obligation improves.
        return after > before and self.v150_grounded_gain(old, new, records, properties)

    @staticmethod
    def v150_facade_preserves_observations(old, new):
        if not old or old.get("load_error") or not new or new.get("load_error"):
            return False
        if old.get("v150_receipt_context") != new.get("v150_receipt_context"):
            return False
        for field in ("checks", "observations"):
            before = old.get(field, [])
            after = new.get(field, [])
            if len(before) != len(after):
                return False
            for a, b in zip(before, after, strict=False):
                error = a.get("error") or {}
                if error.get("skipped") or (b.get("error") or {}).get("skipped"):
                    return False
                if error.get("type") == "TypeError":
                    if b.get("ok") is not True and not (
                        b.get("ok") is False and (b.get("error") or {}).get("type") == "TypeError"
                    ):
                        return False
                    continue
                if a.get("ok") is not b.get("ok"):
                    return False
                key = "actual_repr" if field == "checks" else "result_key"
                if a.get(key) != b.get(key):
                    return False
                if field == "observations" and a.get("mutated_arguments") != b.get(
                    "mutated_arguments"
                ):
                    return False
        before_properties, after_properties = old.get("properties", []), new.get("properties", [])
        if len(before_properties) != len(after_properties):
            return False
        if any(
            a.get("ok") is True and b.get("ok") is not True
            for a, b in zip(before_properties, after_properties, strict=False)
        ):
            return False
        return True

    @staticmethod
    def v150_observations(report, inputs):
        evidence = []
        if not report:
            return evidence
        live = [i for i in inputs if i.get("domain_status") != "excluded"]
        for c, row in enumerate(report["candidates"]):
            if row.get("load_error"):
                continue
            for i, event in enumerate(row["observations"]):
                if i >= len(live):
                    continue
                base = dict(
                    id=f"d{c}:i{i}",
                    argument_literals=repr(live[i]["args"]),
                    domain_status=live[i]["domain_status"],
                )
                if event.get("ok") is True and len(event.get("result_repr", "")) <= 400:
                    try:
                        base.update(
                            result=_read_literal(event["result_repr"]),
                            result_literal=event["result_repr"],
                        )
                    except (ValueError, TypeError, SyntaxError, RecursionError):
                        continue
                elif ClariBridge.v150_usable_error(event):
                    base["exception"] = event.get("error")
                else:
                    continue
                evidence.append(base)
        return evidence

    def v150_question(self, session, anchor, report, inputs, records):
        measured = self.v150_observations(report, inputs)
        eligible = _contract_witnesses(anchor, report, inputs, measured)
        prompt = session.packet["original_prompt"]
        packet = dict(
            session.packet,
            anchor_source=anchor,
            measured_observations=measured,
            anchor_observations=eligible,
            allowed_witness_ids=[item["id"] for item in eligible],
            visible_examples=records,
        )
        raw = session.request("behavior_question", packet)
        data, rejected = _v1508_contract_fields(raw, eligible, prompt)
        keys = ("input_summary", "result_summary", "operation_summary")
        retained = sum(key in data for key in keys)
        decisions, decision_rejected = _v1508_decisions(raw, prompt)
        recovered = 0
        if retained < 3:
            revision = session.request(
                "question_revision", dict(packet, rejected_proposal=raw, usable_fields=data)
            )
            fixed, fixed_rejected = _v1508_contract_fields(revision, eligible, prompt)
            rejected.extend(fixed_rejected)
            for key in keys:
                if key not in data and key in fixed:
                    data[key] = fixed[key]
                    recovered += 1
            if not data["witness_id"]:
                data["witness_id"] = fixed["witness_id"]
            if not data["conflict_spans"]:
                data["conflict_spans"] = fixed["conflict_spans"]
            if not decisions:
                decisions, more_rejected = _v1508_decisions(revision, prompt)
                decision_rejected.extend(more_rejected)
        witness = next((item for item in eligible if item["id"] == data["witness_id"]), None)
        witness = witness or (eligible[0] if eligible else None)
        fallback = _fallback_contract(anchor, session.packet["entry_point"], witness)
        missing = [key for key in keys if key not in data]
        for key in missing:
            data[key] = fallback[key]
        question, binding = _v1508_render_contract(
            data,
            session.packet["entry_point"],
            witness,
            _v129_source_sha(anchor),
            decisions,
            self.config.get("illustrative_example", False),
        )
        session.events.append(
            dict(
                role="contract_render",
                status="complete" if not missing else "partial_summary_fallback",
                measured_example=bool(witness) and not decisions,
                summary_fields_retained=retained,
                summary_fields_recovered=recovered,
                summary_fields_fallback=len(missing),
                auxiliary_rejections=rejected,
                decision_count=len(decisions),
                decision_rejections=decision_rejected,
                decision_list_observed=isinstance(raw, dict)
                and isinstance(raw.get("decisions"), list),
                decision_proposed_count=len(raw["decisions"])
                if isinstance(raw, dict) and isinstance(raw.get("decisions"), list)
                else None,
                eligible_witness_count=len(eligible),
                witness_status="selected" if witness else "no_eligible_witness",
                question_sha256=binding["question_sha256"],
                protocol=binding["protocol"],
            )
        )
        return question, binding

    def v150_arbitrate(self, session, records, properties, inputs, reason, report=None):
        if session.arbitrated or not session.packet["actual_answer"] or session.remaining() <= 1:
            return records, properties, inputs
        session.arbitrated = True
        decision = session.request(
            "evidence_arbitration",
            dict(
                session.packet,
                reason=reason,
                records=records,
                properties=properties,
                inputs=inputs,
                measured_execution=report,
                judgment_is_fallible=True,
            ),
            reserve=1,
        )
        records, properties, inputs = apply_arbitration(
            decision.get("decisions", []),
            session.packet["actual_answer"],
            records,
            properties,
            inputs,
        )
        return (
            records,
            [
                p
                for p in properties
                if p.get("status", "active") == "active" and p.get("tier") in {"R", "P"}
            ],
            inputs,
        )

    def v150_failures(self, row, records, properties, inputs):
        result = [
            dict(kind="io", obligation=r, measured=e)
            for r, e in zip(records, row["checks"], strict=False)
            if self.v150_usable_error(e)
        ]
        result += [
            dict(kind="property", obligation=p, observations=row["observations"])
            for p, value in zip(
                properties, self.v150_property_values(row, properties), strict=False
            )
            if value is False
        ]
        live = [i for i in inputs if i.get("domain_status") != "excluded"]
        result += [
            dict(kind="domain_runtime", input=item, measured=e)
            for item, e in zip(live, row["observations"], strict=False)
            if item.get("domain_status") == "confirmed" and self.v150_usable_error(e)
        ]
        if row.get("load_error") and not row["load_error"].get("skipped"):
            result.append(dict(kind="entry_load", measured=row["load_error"]))
        return result

    def v150_repair(self, session, code, row, records, properties, inputs):
        for _ in range(2):
            if session.remaining() <= 1 or session.executions >= session.exec_cap - 1:
                break
            failures = self.v150_failures(row, records, properties, inputs)
            if not failures or not self.v150_grounded_failure(row, records, properties):
                break
            candidate = session.code(
                session.request(
                    "execution_patch",
                    dict(
                        session.packet,
                        current_source=code,
                        source_backed_failures=failures,
                        evidence_records=records,
                        properties=properties,
                        inputs=inputs,
                        measured_execution=row,
                    ),
                    reserve=1,
                )
            )
            if not candidate:
                break
            trial = session.execute([candidate], records, inputs, properties)
            if not trial or not self.v150_can_accept(
                row, trial["candidates"][0], records, properties, inputs
            ):
                break
            code, row = candidate, trial["candidates"][0]
        return code, row

    def v150_campaign_plan(self, session):
        if not self.config.get("planner") or session.remaining() <= 5:
            return []
        proposal = session.request("implementation_plan", dict(session.packet))
        routes = []
        for item in proposal.get("routes", [])[:3]:
            if not isinstance(item, dict):
                continue
            span, outline = item.get("source_span"), item.get("outline")
            if (
                isinstance(outline, str)
                and outline.strip()
                and (
                    quote(session.packet["actual_answer"], span)
                    or quote(session.packet["original_prompt"], span)
                )
            ):
                routes.append(
                    dict(source_span=span, outline=outline[:1600], authority="fallible_plan")
                )
        return routes

    def v150_campaign_choose(self, session, pool, report, records, properties, inputs):
        count = self.config.get("reviewers", 0)
        if not count or not report or len(pool) < 2 or session.remaining() <= count:
            return None
        scores = [self.v150_score(row, records, properties, inputs) for row in report["candidates"]]
        eligible = {i for i, value in enumerate(scores) if value == max(scores)}
        if len(eligible) < 2:
            return None
        packet = dict(
            session.packet,
            candidates=[dict(candidate_index=i, source=pool[i]) for i in sorted(eligible)],
            measured_execution=report,
            evidence_records=records,
            properties=properties,
        )
        perspectives = (
            "central input-to-output operation",
            "argument and return contract",
            "conditions, negation and exceptional cases",
        )
        replies = session.batch(
            [
                ("candidate_review", dict(packet, review_focus=perspectives[i % 3]))
                for i in range(count)
            ]
        )

        def valid(item):
            return (
                isinstance(item, dict)
                and type(item.get("candidate_index")) is int
                and item["candidate_index"] in eligible
                and isinstance(item.get("reason"), str)
                and item["reason"].strip()
                and isinstance(item.get("source_span"), str)
                and len(item["source_span"].strip()) >= 8
                and (
                    quote(session.packet["actual_answer"], item["source_span"])
                    or quote(session.packet["original_prompt"], item["source_span"])
                )
            )

        votes = [item for item in replies if valid(item)]
        if (
            self.config.get("arbiter")
            and len({item["candidate_index"] for item in votes}) > 1
            and session.remaining() > 1
        ):
            decision = session.request(
                "candidate_review",
                dict(
                    packet,
                    review_focus="resolve the cited source conflict",
                    independent_reviews=votes,
                ),
            )
            return decision["candidate_index"] if valid(decision) else None
        if not votes:
            return None
        counts = {i: sum(item["candidate_index"] == i for item in votes) for i in eligible}
        best = max(counts, key=counts.get)
        # A multi-review decision requires an absolute majority, not mere plurality.
        return best if count == 1 or counts[best] > count / 2 else None

    def v150_campaign_repair(self, session, pool, report, records, properties, inputs, code, row):
        if not self.config.get("branch_repair"):
            return self.v150_repair(session, code, row, records, properties, inputs)
        order = sorted(
            range(len(pool)),
            key=lambda i: (
                self.v150_score(report["candidates"][i], records, properties, inputs),
                -i,
            ),
            reverse=True,
        )[:2]
        for index in order:
            original = report["candidates"][index]
            if not self.v150_grounded_failure(original, records, properties):
                continue
            revised, measured = self.v150_repair(
                session, pool[index], original, records, properties, inputs
            )
            pool[index], report["candidates"][index] = revised, measured
            report["source_sha_by_index"][index] = _v129_source_sha(revised)
        chosen = self.v150_campaign_choose(session, pool, report, records, properties, inputs)
        if chosen is None:
            chosen = self.v150_rank(report, records, properties, inputs)
        return pool[chosen], report["candidates"][chosen]

    def run(self, env, problem):
        session = _V129Session(self, env)
        session.packet = dict(
            original_prompt=problem["prompt"],
            entry_point=problem["entry_point"],
            question="",
            actual_answer="",
        )
        entry = session.packet["entry_point"]
        records = public_records(session.packet["original_prompt"], entry)
        tasks = [
            ("initial_builder", dict(session.packet, construction_view=view))
            for view in V129_PROMPTS["construction_views"]["initial"][:3]
        ]
        tasks.append(("observation_inputs", dict(session.packet, phase="before_answer")))
        replies = session.batch(tasks)
        drafts = list(dict.fromkeys(c for c in (session.code(r) for r in replies[:-1]) if c))
        placeholder = f'def {entry}(*args, **kwargs):\n    raise NotImplementedError("No usable implementation was returned")'
        drafts = drafts or [placeholder]
        seed = [
            dict(
                id=f"public-{i}",
                args=r["args"],
                source="prompt",
                source_span=r["source_span"],
                domain_status="confirmed",
            )
            for i, r in enumerate(records)
        ]
        inputs = parse_inputs(replies[-1], session.packet, previous=seed)
        active = self.v150_active(records)
        initial = session.execute(drafts, active, inputs)
        anchor_index = self.v150_rank(initial, active, inputs=inputs) if initial else 0
        anchor = drafts[anchor_index]
        if env.can_ask():
            question, session.binding = self.v150_question(
                session, anchor, initial, inputs, records
            )
            session.packet["question"] = question + "\n\n" + V129_PROMPTS["sdk_question_suffix"]
            try:
                answer = env.ask_human(session.packet["question"])
                if (
                    isinstance(answer, str)
                    and answer.strip()
                    and answer.strip()
                    not in {"Irrelevant question.", "Cannot answer the given question."}
                ):
                    session.packet["actual_answer"] = answer
            except Exception as exc:
                session.events.append(
                    dict(role="clarification", status="unavailable", exception=v150_type_name(exc))
                )
        properties = []
        pool, report = drafts, initial
        fallback = anchor
        if session.packet["actual_answer"]:
            mapped = session.request(
                "reply_map",
                dict(
                    session.packet,
                    question_binding=session.binding,
                    original_examples=records,
                    previous_inputs=inputs,
                ),
            )
            compiled = compile_reply(
                mapped,
                session.packet["actual_answer"],
                session.packet["original_prompt"],
                entry,
                records,
                session.binding,
            )
            session.events.append(
                dict(
                    role="v15010_property_compile",
                    status="measured",
                    **compiled["property_protocol"],
                )
            )
            records, properties = compiled["records"], compiled["properties"]
            session.packet.update(
                reply_effect=compiled["effect"],
                core_intent=mapped.get("core_intent", ""),
                advisory_rules=compiled.get("advisory", []),
                reply_coverage=compiled["reply_coverage"],
            )
            inputs = parse_inputs(
                {}, session.packet, previous=inputs, updates=compiled.get("input_updates", [])
            )
            domain_change = (
                any(
                    i.get("action") in {"exclude", "uncertain"}
                    for i in compiled.get("input_updates", [])
                )
                or compiled["effect"] == "redefine"
                and _v129_domain_evidence(
                    session.packet["actual_answer"], session.packet["actual_answer"]
                )
            )
            if compiled.get("disputes") or domain_change:
                records, properties, inputs = self.v150_arbitrate(
                    session,
                    records,
                    properties,
                    inputs,
                    dict(
                        kind="source_or_domain_scope",
                        proposed_changes=compiled["disputes"],
                        domain_changed=bool(domain_change),
                    ),
                )
            routes = self.v150_campaign_plan(session)
            common = dict(session.packet, evidence_records=records, properties=properties)
            if routes:
                common["implementation_routes"] = routes
            tasks = [
                ("preserving_builder", dict(common, anchor_source=anchor)),
                ("reconstruction_builder", dict(common)),
                ("boundary_builder", dict(common, confirmed_domain=inputs)),
                ("observation_inputs", dict(common, phase="after_answer", previous_inputs=inputs)),
            ]
            extra = max(0, min(2, int(self.config.get("fresh_candidates", 3)) - 3))
            for direction in (
                "derive a compact defining invariant",
                "use a direct transparent algorithm with explicit argument roles",
            )[:extra]:
                tasks.insert(
                    -1, ("reconstruction_builder", dict(common, search_direction=direction))
                )
            replies = session.batch(tasks)
            fresh = list(dict.fromkeys(c for c in (session.code(r) for r in replies[:-1]) if c))
            inputs = parse_inputs(replies[-1], session.packet, previous=inputs)
            pool = fresh or [anchor]
            fallback = pool[0]
            session.events.append(
                dict(
                    role="candidate_lineage",
                    status="answer_conditioned" if fresh else "all_fresh_invalid_fallback",
                    count=len(fresh),
                )
            )
            active = self.v150_active(records)
            report = session.execute(pool, active, inputs, properties)
            if report and not session.arbitrated and session.executions < session.exec_cap - 1:
                suspect = [
                    r["id"]
                    for j, r in enumerate(active)
                    if r.get("source") == "prompt"
                    and all(
                        self.v150_loaded(row) and row["checks"][j].get("ok") is False
                        for row in report["candidates"]
                    )
                ]
                if suspect:
                    previous_context = report["context_key"]
                    records, properties, inputs = self.v150_arbitrate(
                        session,
                        records,
                        properties,
                        inputs,
                        dict(kind="inspect_checks_not_vote", ids=suspect),
                        report,
                    )
                    active = self.v150_active(records)
                    if (
                        self.v150_context_key(active, inputs, properties) != previous_context
                        and session.executions < session.exec_cap - 1
                    ):
                        report = session.execute(pool, active, inputs, properties)
                        session.completed_unknown = True
            if report:
                winner = self.v150_rank(report, active, properties, inputs)
                row = report["candidates"][winner]
                hard_values = [
                    row["checks"][i].get("ok") for i, r in enumerate(active) if r.get("tier") == "H"
                ]
                hard_values += [
                    v
                    for p, v in zip(
                        properties, self.v150_property_values(row, properties), strict=False
                    )
                    if p.get("source") == "answer"
                ]
                if (
                    any(v is None for v in hard_values)
                    and not session.completed_unknown
                    and session.executions < session.exec_cap - 1
                ):
                    session.completed_unknown = True
                    report = session.execute(pool, active, inputs, properties, force=True)
                if (
                    session.remaining() > 1
                    and session.executions < session.exec_cap - 1
                    and (
                        all(
                            self.v150_failures(row, active, properties, inputs)
                            for row in report["candidates"]
                        )
                        or self.config.get("disagreement_rescue")
                        and _campaign_disagreement(report, inputs)
                    )
                ):
                    rescue = session.code(
                        session.request(
                            "obligation_rescue",
                            dict(
                                session.packet,
                                evidence_records=records,
                                properties=properties,
                                measured_execution=report,
                                inputs=inputs,
                            ),
                            reserve=1,
                        )
                    )
                    if rescue and rescue not in pool:
                        trial = session.execute([rescue], active, inputs, properties)
                        if trial:
                            pool.append(rescue)
                            report["candidates"].append(trial["candidates"][0])
                            report["source_sha_by_index"].append(_v129_source_sha(rescue))
            records = active
        else:
            records = active
        if report:
            chosen = self.v150_campaign_choose(session, pool, report, records, properties, inputs)
            if chosen is None:
                chosen = self.v150_rank(report, records, properties, inputs)
            code, row = pool[chosen], report["candidates"][chosen]
        else:
            code, row = fallback, None
        if row:
            code, row = self.v150_campaign_repair(
                session, pool, report, records, properties, inputs, code, row
            )
        if row and self.v150_grounded_failure(row, records, properties) and session.remaining() > 0:
            review = session.request(
                "reply_compliance_review",
                dict(
                    session.packet,
                    current_source=code,
                    evidence_records=records,
                    properties=properties,
                    measured_execution=row,
                ),
            )
            span = review.get("source_span")
            cited = quote(session.packet["actual_answer"], span) or quote(
                session.packet["original_prompt"], span
            )
            if review.get("decision") == "propose" and cited and review.get("reason"):
                proposed = extract_code(review.get("replacement_source"), entry)
                if proposed and session.executions < session.exec_cap - 1:
                    trial = session.execute([artifact(proposed)], records, inputs, properties)
                    if trial and self.v150_can_accept(
                        row, trial["candidates"][0], records, properties, inputs
                    ):
                        code, row = artifact(proposed), trial["candidates"][0]
        facade = _v132_length_facade(code, entry)
        if facade != code and row and session.executions < session.exec_cap:
            trial = session.execute([facade], records, inputs, properties)
            if trial and self.v150_facade_preserves_observations(row, trial["candidates"][0]):
                code = facade
        return "```python\n" + code + "\n```"
