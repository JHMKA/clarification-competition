import hashlib
import importlib.util
import inspect
import json
import os
import sys
import traceback

import fire
from rich.console import Console
from rich.panel import Panel
from tqdm import tqdm

from clarify.baselines.base import ClarificationAlgorithmBase
from clarify.data import load_split, preprocess_benchmark
from clarify.env import ClarificationConfiguration, ClarificationEnvironment, test_context
from clarify.utils import BatchParallelProcessor, BatchSequentialProcessor

console = Console()

CLARIFICATION_PATH = "data/clarifications.jsonl"

DEFAULT_DATASET_PATH = "data/mbpp_demo_test.jsonl"

DEFAULT_SPLITS = {"train": "data/splits/train.txt", "val": "data/splits/validation.txt"}


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
        for name, obj in inspect.getmembers(module, inspect.isclass)
        if issubclass(obj, ClarificationAlgorithmBase)
        and obj is not ClarificationAlgorithmBase
        and obj.__module__ == module_name  # exclude re-imported subclasses from elsewhere
        and not name.startswith("_")  # exclude candidates prefixed with "_"
    ]

    if not candidates:
        raise ValueError(f"No ClarificationAlgorithmBase subclass defined in {path_to_algorithm}.")
    if len(candidates) > 1:
        raise ValueError(
            f"Expected exactly one ClarificationAlgorithmBase subclass in "
            f"{path_to_algorithm}, found: {[c.__name__ for c in candidates]}"
        )

    return candidates[0]


# Hooks ---------------


def llm_console_hook(message: dict[str, str]):
    if message["role"] == "user":
        console.print(Panel(message["content"], title="Prompt", border_style="cyan"))
    if message["role"] == "assistant":
        console.print(Panel(message["content"], title="Model Response", border_style="purple"))


def clarification_console_hook(message: dict[str, str]):
    if message["role"] == "user":
        console.print(Panel(message["content"], title="Question", border_style="green"))
    if message["role"] == "assistant":
        console.print(Panel(message["content"], title="Clarification", border_style="orange3"))


# ---------------------


class SimulationFunction:
    _algorithm_cache: dict[str, ClarificationAlgorithmBase] = {}

    def __init__(self, **config):
        self.config = config

    def _get_algorithm(self) -> ClarificationAlgorithmBase:
        path = self.config["clarification_algorithm_path"]
        if path not in self._algorithm_cache:
            cls = _load_clarification_algorithm(path)
            kwargs = self.config.get("clarification_algorithm_kwargs", {})
            self._algorithm_cache[path] = cls(kwargs)
        return self._algorithm_cache[path]

    def batch(self, environment_definitions):
        clarification_algorithm = self._get_algorithm()
        algorithm_name = clarification_algorithm.__class__.__name__
        if not hasattr(clarification_algorithm, "batch_run"):
            raise ValueError(f"Batch processing is not supported by {algorithm_name}")

        envs, problem_definitions, results = [], [], []

        for environment_definition in environment_definitions:
            envs.append(
                ClarificationEnvironment(
                    self.config["environment_config"],
                    environment_definition,
                    llm_api_hook=llm_console_hook,
                    clarification_api_hook=clarification_console_hook,
                )
            )

            problem_definitions.append(
                {
                    "prompt": environment_definition["prompt"],
                    "entry_point": environment_definition.get("entry_point", "unknown"),
                }
            )

            results.append(
                {
                    "task_id": environment_definition["task_id"],
                    "prompt": environment_definition["prompt"],
                }
            )

        try:
            with console.status(f"Run {algorithm_name} ({len(envs)} instances)..."):
                with test_context():
                    prompt_results = clarification_algorithm.batch_run(envs, problem_definitions)

                for i, prompt_result in enumerate(prompt_results):
                    environment = envs[i]
                    results[i].update(
                        {
                            "algorithm": algorithm_name,
                            "model": self.config["environment_config"].language_model,
                            "algorithm_path": self.config["clarification_algorithm_path"],
                            "prompt_result": prompt_result,
                            "need_clarification": len(
                                environment_definition.get("clarifications", [None])
                            )
                            > 0,
                            "clarification_history": environment.history,
                            "prompt_cost": environment.prompt_cost,
                            "clarification_cost": environment.clarification_cost or 0.0,
                        }
                    )

                return results
        except Exception as e:
            if self.config["fail_on_exception"]:
                raise

            traceback.print_exc()
            for result in results:
                result["prompt_result"] = f"[EXCEPTION] {e}"
            return results

    def __call__(self, environment_definition):
        environment = ClarificationEnvironment(
            self.config["environment_config"],
            environment_definition,
            llm_api_hook=llm_console_hook,
            clarification_api_hook=clarification_console_hook,
        )

        problem_definition = {
            "prompt": environment_definition["prompt"],
            "entry_point": environment_definition.get("entry_point", "unknown"),
        }

        result = {
            "task_id": environment_definition["task_id"],
            "prompt": environment_definition["prompt"],
        }

        try:
            clarification_algorithm = self._get_algorithm()
            algorithm_name = clarification_algorithm.__class__.__name__
            with test_context(), console.status(f"Run {algorithm_name}..."):
                prompt_result = clarification_algorithm.run(environment, problem_definition)

            result.update(
                {
                    "algorithm": algorithm_name,
                    "model": self.config["environment_config"].language_model,
                    "algorithm_path": self.config["clarification_algorithm_path"],
                    "prompt_result": prompt_result,
                    "need_clarification": len(environment_definition.get("clarifications", [None]))
                    > 0,
                    "clarification_history": environment.history,
                    "prompt_cost": environment.prompt_cost,
                    "clarification_cost": environment.clarification_cost or 0.0,
                }
            )

            return result

        except Exception as e:
            if self.config["fail_on_exception"]:
                raise

            traceback.print_exc()
            result["prompt_result"] = f"[EXCEPTION] {e}"
            return result


def main(
    clarify_py: str,
    dataset_path: str = DEFAULT_DATASET_PATH,
    output_path: str = "data/mbpp_demo_test_clarify_output.jsonl",
    batch_size: int = 1,
    max_workers: int = 1,
    max_clarification_turns: int = 1,
    language_model: str = "openai/gpt-4.1-mini",
    temperature: float = 0.7,
    clarification_model: str | None = None,
    fail_on_exception: bool = False,
    max_prompt_budget: float = 1.0,
    max_clarification_budget: float = 1.0,
    num_samples: int = 1,
    split: str | None = None,
    decision_model: str | None = None,
    **kwargs,
):
    batch_size = max(batch_size, max_workers)

    environment_config = ClarificationConfiguration(
        max_clarification_turns=max_clarification_turns,
        language_model=language_model,
        temperature=temperature,
        clarification_model=clarification_model,
        decision_model=decision_model,
        max_clarification_budget=max_clarification_budget,
        max_prompt_budget=max_prompt_budget,
    )

    # Load clarification algorithm and use kwargs as config options
    clarification_algorithm_class = _load_clarification_algorithm(clarify_py)

    algorithm_name = clarification_algorithm_class.__name__
    print(f"Loaded `{algorithm_name}` clarification algorithm...")

    # Load data ----------------------

    if split:
        if dataset_path != DEFAULT_DATASET_PATH:
            print(f"WARNING: split '{split}' is overwritting your dataset path.")
        split = DEFAULT_SPLITS.get(split, split)
        benchmark = load_split(split)
    else:
        with open(dataset_path, "r") as lines:
            benchmark = [json.loads(line) for line in lines]

        benchmark = list(preprocess_benchmark(benchmark).values())

    print(f"Loaded {len(benchmark)} instances...")

    if os.path.exists(CLARIFICATION_PATH):
        clarification_index = {}
        with open(CLARIFICATION_PATH, "r") as lines:
            for line in lines:
                example = json.loads(line)
                clarification_index[(example["task_id"], example["prompt"])] = example[
                    "clarifications"
                ]

        for example in benchmark:
            if (example["task_id"], example["prompt"]) in clarification_index:
                example["clarifications"] = clarification_index[
                    (example["task_id"], example["prompt"])
                ]

    simulation_function = SimulationFunction(
        clarification_algorithm_path=clarify_py,
        clarification_algorithm_kwargs=kwargs,
        environment_config=environment_config,
        fail_on_exception=fail_on_exception,
    )

    def _duplicate(dataset, k: int = 1):
        for example in dataset:
            for i in range(k):
                example["sample_id"] = i
                yield example

    def batched_iterator(k: int = 1):
        current_batch = []
        for example in _duplicate(benchmark, k):
            current_batch.append(example)
            if len(current_batch) >= batch_size:
                yield current_batch
                current_batch = []

        if len(current_batch) > 0:
            yield current_batch

    if max_workers > 1:
        batched_worker = BatchParallelProcessor(simulation_function, max_workers=max_workers)
    else:
        batched_worker = BatchSequentialProcessor(simulation_function)

    prompt_cost, clarification_cost = 0.0, 0.0
    try:
        with open(output_path, "w") as o, tqdm(total=len(benchmark)) as pbar:
            for task_batch in batched_iterator(k=num_samples):
                try:
                    results = simulation_function.batch(task_batch)
                except ValueError:
                    results = batched_worker(task_batch)

                for result in results:
                    o.write(json.dumps(result, default=str) + "\n")

                    prompt_cost += result.get("prompt_cost", 0.0)
                    clarification_cost += result.get("clarification_cost", 0.0)

                    pbar.set_description(
                        f"{algorithm_name} | Prompt ($): {prompt_cost + clarification_cost:.4f}, Clarification ($): {clarification_cost:.4f}"
                    )
                    pbar.update(1)

    finally:
        if batched_worker:
            batched_worker.close()


if __name__ == "__main__":
    fire.Fire(main)
