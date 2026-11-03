import csv
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import fire
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from tqdm import tqdm

from clarify.data import load_split, preprocess_benchmark
from clarify.runtime import init_evalplus_evaluator
from clarify.utils import BatchParallelProcessor, BatchSequentialProcessor

console = Console()


# ---------------------

DEFAULT_DATASET_PATH = "data/mbpp_demo_test.jsonl"

DEFAULT_SPLITS = {"train": "data/splits/train.txt", "val": "data/splits/validation.txt"}

PUBLIC_LEADERBOARD = Path("docs/data/leaderboard.csv")
PRIVATE_LEADERBOARD = Path("docs/data/private-leaderboard.csv")


class EvaluationFunction:
    def __init__(self, **config):
        self.config = config

    def __call__(self, result):
        evaluator = result["evaluator"]
        del result["evaluator"]

        prompt_result = result["prompt_result"]
        if "```python" not in prompt_result:
            prompt_result = f"```python\n{prompt_result.strip()}\n```"

        # Execute
        with console.status(f"Run tests for {result['task_id']}"):
            success, status = evaluator.evaluate(None, prompt_result)

        if success:
            console.print(
                Panel(
                    prompt_result, title=f"✅ Success ({result['task_id']})", border_style="green"
                )
            )
        else:
            output = f"{prompt_result}\n\nTest Result:\n{status}"
            console.print(
                Panel(output, title=f"❌ Failure ({result['task_id']})", border_style="red")
            )

        result["success"], result["test_result"] = success, status
        return result


# Statistics ---------


def _print_test_overview(results):
    clarify_map = {"3": "✅", "2": "🌀", "1": "☠️"}
    algorithm = None
    for result in results:
        algorithm = result.get("algorithm", None)

    if algorithm:
        table = Table(title=f"Test Results ({algorithm})")
    else:
        table = Table(title="Test Results")

    table.add_column("Task ID")
    table.add_column("Status", justify="center")
    table.add_column("Clarify?", justify="center")
    table.add_column("Output", no_wrap=True)

    for result in sorted(results, key=lambda x: x["task_id"]):
        output = result["test_result"]

        clarification = "❌"
        if result.get("clarification_history", []):
            clarification = max(e[2] for e in result["clarification_history"])
            clarification = clarify_map.get(clarification, clarification)

        table.add_row(
            result["task_id"],
            "✅" if result["success"] else ("⏲️" if "timeout" in output else "❌"),
            clarification,
            output[:40] + "..." if len(output) > 40 else output,
        )

    console.print(table)


def _turn_discounted_sucess(results, base: int = 10):
    turn_discounted_sucess = 0.0
    for result in results:
        clarification_length = len(result.get("clarification_history", []))
        if result["success"]:
            turn_discounted_sucess += math.log(base) / math.log(base + clarification_length)

    return turn_discounted_sucess / len(results)


def _normalized_discounted_cumulative_gain(results, base: int = 2):
    nDCG = 0.0
    for result in results:
        indicator = [e[2] == "3" for e in result.get("clarification_history", [])]

        discounted_cumulative_gain = 0.0
        idealized_cumulative_gain = 0.0
        for turn, high_quality in enumerate(indicator):
            gain = math.log(base) / math.log(base + turn)
            if high_quality:
                discounted_cumulative_gain += gain
            idealized_cumulative_gain += gain

        if idealized_cumulative_gain > 0:
            nDCG += discounted_cumulative_gain / idealized_cumulative_gain

    return nDCG / len(results)


def print_statistics(output_path):
    with open(output_path, "r") as lines:
        results = [json.loads(line) for line in lines]

    _print_test_overview(results)

    table = Table(title="Run Statistics")
    table.add_column("Metric")
    table.add_column("Results")

    total = len(results)
    table.add_row("Total", str(total))

    table.add_row("Turn Discounted Success", f"{_turn_discounted_sucess(results):.4f}")
    table.add_row("nDCG", f"{_normalized_discounted_cumulative_gain(results):.4f}")

    pass_at_1 = sum(r["success"] for r in results)
    table.add_row("Pass@1", f"{100 * pass_at_1 / total:.2f}")

    clarification_rate = sum(len(r.get("clarification_history", [])) > 0 for r in results)
    table.add_row("Clarification rate", f"{100 * clarification_rate / total:.2f}%")

    overask_result = [r for r in results if not r.get("need_clarification", True)]
    overasking_rate = 0.0
    if overask_result:
        overasking_rate = sum(
            len(r.get("clarification_history", [])) > 0 for r in overask_result
        ) / len(overask_result)

    table.add_row("Overasking rate", f"{100 * overasking_rate:.2f}%")

    if clarification_rate > 0:
        high_quality_clarification = sum(
            all(e[2] == "3" for e in r.get("clarification_history", []))
            for r in results
            if len(r.get("clarification_history", [])) > 0
        )
        table.add_row(
            "High quality clarification",
            f"{100 * high_quality_clarification / clarification_rate:.2f}%",
        )

        clarification_length = sum(
            sum(len(e[3]) for e in r.get("clarification_history", [])) for r in results
        )
        table.add_row(
            "Average clarification length", f"{clarification_length / clarification_rate:.2f}"
        )

    console.print(table)


# --------------------


def _find_repo_root() -> Path:
    """Find the repository root containing docs/data/leaderboard.csv."""
    starts = (Path(__file__).resolve().parent, Path.cwd().resolve())
    seen: set[Path] = set()
    for start in starts:
        for candidate in (start, *start.parents):
            if candidate in seen:
                continue
            seen.add(candidate)
            if (candidate / PUBLIC_LEADERBOARD).is_file():
                return candidate
    raise FileNotFoundError(
        f"Could not find {PUBLIC_LEADERBOARD}. Run from inside the competition repository."
    )


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        return fieldnames, [dict(row) for row in reader]


def _compute_output_row(results):
    algorithms = set(r["algorithm"] for r in results)
    assert len(algorithms) == 1, f"Expected only one algorithm in results, but got {algorithms}"
    models = set(r["model"] for r in results)
    assert len(models) == 1, f"Expected only one model in results, but got {models}"
    paths = set(r["algorithm_path"] for r in results)
    assert len(paths) == 1, f"Expected only one algorithm path in results, but got {paths}"

    algorithm, path, model = next(iter(algorithms)), next(iter(paths)), next(iter(models))
    baseline = "baseline" in path

    total = len(results)
    tds = _turn_discounted_sucess(results)
    ndcg = _normalized_discounted_cumulative_gain(results)

    pass_at_1 = sum(r["success"] for r in results) / total
    clarification_rate = sum(len(r.get("clarification_history", [])) > 0 for r in results) / total

    overask_result = [r for r in results if not r.get("need_clarification", False)]
    overasking_rate = 0.0
    if overask_result:
        overasking_rate = sum(
            len(r.get("clarification_history", [])) > 0 for r in overask_result
        ) / len(overask_result)

    prompt_cost = sum(r.get("prompt_cost", 0.0) for r in results) / total

    return {
        "track": "main",
        "algorithm": algorithm,
        "model": model,
        "tds": tds,
        "pass_at_1": pass_at_1,
        "ndcg": ndcg,
        "clarification_rate": clarification_rate,
        "over_asking_rate": overasking_rate,
        "avg_cost_usd": prompt_cost,
        "is_baseline": baseline,
        "submission_url": f"https://github.com/serval-uni-lu/clarification-competition/tree/main/{path}",
    }


def submit_to_benchmark(output_path, split=None, team=None, trusted=False):
    if split not in {"val", "test"}:
        print("> Split must be either 'val' or 'test'; leaderboard was not updated.")
        return

    split = DEFAULT_SPLITS.get(split, split)
    benchmark = load_split(split)
    benchmark_index = set((ex["task_id"], ex["prompt"]) for ex in benchmark)

    with open(output_path, "r") as lines:
        results = [json.loads(line) for line in lines]

    result_index = set((result["task_id"], result["prompt"]) for result in results)
    difference = benchmark_index.difference(result_index)
    if difference:
        print(f"> Results are missing entries for {difference}; leaderboard was not updated.")
        return

    if len(result_index) != len(benchmark_index):
        print("> The result set does not match the benchmark; leaderboard was not updated.")
        return

    repo_root = _find_repo_root()
    public_path = repo_root / PUBLIC_LEADERBOARD
    target_path = repo_root / (PUBLIC_LEADERBOARD if "val" in split else PRIVATE_LEADERBOARD)

    fieldnames, _ = _read_csv(public_path)

    if target_path.exists():
        _, rows = _read_csv(target_path)
    else:
        rows = []

    result = _compute_output_row(results)

    algorithm = str(result.get("algorithm", "")).strip()
    model = str(result.get("model", "")).strip()

    evaluated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    insert_defaults = {field: "" for field in fieldnames}
    insert_defaults.update(
        {
            "track": "main",
            "algorithm": algorithm,
            "model": model,
            "date": evaluated_at[:10],
            "team": team if isinstance(team, str) else "",
            "evaluated_at": evaluated_at,
            "is_baseline": "false",
            "confirmed": trusted,
        }
    )

    normalized_result = {
        key: (str(value).lower() if isinstance(value, bool) else str(value))
        for key, value in result.items()
        if key in fieldnames and value is not None
    }
    normalized_result.update(
        {
            "algorithm": algorithm,
            "model": model,
            "date": evaluated_at[:10],
            "evaluated_at": evaluated_at,
        }
    )

    updated = False
    for index, row in enumerate(rows):
        if row.get("algorithm", "").strip() == algorithm and row.get("model", "").strip() == model:
            merged = {field: row.get(field, "") for field in fieldnames}
            merged.update(normalized_result)
            rows[index] = merged
            updated = True
            break

    if not updated:
        insert_defaults.update(normalized_result)
        rows.append(insert_defaults)

    with target_path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print("Success! Wrote to the leaderboard.")


def main(
    benchmark_path: str = "data/mbpp_demo_test.jsonl",
    generation_path: str = "data/mbpp_demo_test_clarify_output.jsonl",
    output_path: str = "data/mbpp_demo_test_clarify_output_results.jsonl",
    batch_size: int = 1,
    max_workers: int = 1,
    force_rerun: bool = False,
    split: str | None = None,
    submit: str | bool = False,
    trusted: bool = False,
):
    if os.path.exists(output_path) and not force_rerun:
        print_statistics(output_path)
        if submit:
            submit_to_benchmark(output_path, split, team=submit, trusted=trusted)
        exit(0)

    batch_size = max(batch_size, max_workers)

    # Load data ----------------------
    if split:
        if benchmark_path != DEFAULT_DATASET_PATH:
            print(f"WARNING: split '{split}' is overwritting your benchmark path.")
        split_path = DEFAULT_SPLITS.get(split, split)
        benchmark = load_split(split_path)
    else:
        with open(benchmark_path, "r") as lines:
            benchmark = [json.loads(line) for line in lines]

        benchmark = list(preprocess_benchmark(benchmark).values())

    evaluator = init_evalplus_evaluator(benchmark)
    print(f"Loaded {len(benchmark)} instances...")

    with open(generation_path, "r") as lines:
        results = [json.loads(line) for line in lines]

    for result in results:
        result["evaluator"] = evaluator.partial(result)

    print(f"Loaded {len(results)} result instances...")

    evaluation_function = EvaluationFunction()

    def batched_iterator():
        current_batch = []
        for example in results:
            current_batch.append(example)
            if len(current_batch) >= batch_size:
                yield current_batch
                current_batch = []

        if len(current_batch) > 0:
            yield current_batch

    if max_workers > 1:
        batched_worker = BatchParallelProcessor(evaluation_function, max_workers=max_workers)
    else:
        batched_worker = BatchSequentialProcessor(evaluation_function)

    num_success, total = 0, 0
    try:
        with open(output_path, "w") as o, tqdm(total=len(benchmark)) as pbar:
            for task_batch in batched_iterator():
                results = batched_worker(task_batch)
                for result in results:
                    o.write(json.dumps(result, default=str) + "\n")

                    num_success += 1.0 if result.get("success", False) else 0.0
                    total += 1

                    pbar.set_description(f"Pass@1: {100 * num_success / total:.2f}")
                    pbar.update(1)

    finally:
        if batched_worker:
            batched_worker.close()
        print_statistics(output_path)
        if submit:
            submit_to_benchmark(output_path, split, team=submit, trusted=trusted)


if __name__ == "__main__":
    fire.Fire(main)
