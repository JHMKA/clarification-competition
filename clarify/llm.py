from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)


class LanguageModel:
    """A lightweight language model wrapper over LiteLLM.

    Handles:

    - **Retries** with exponential backoff via LiteLLM's ``num_retries``.
    - **Truncation detection** — logs a warning when ``finish_reason='length'``.
    - **drop_params=True** so unsupported params are silently ignored
      (with a warning logged for transparency).

    Args:
        model: LiteLLM model identifier, e.g. ``"openai/gpt-4.1"`` or ``"anthropic/claude-sonnet-4-6"``.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        num_retries: Number of retries on transient failures (default 3).
        **kwargs: Extra keyword arguments forwarded to ``litellm.completion``
            (e.g. ``top_p``, ``stop``, ``api_key``, ``api_base``).
    """

    def __init__(
        self,
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        num_retries: int = 3,
        **kwargs: Any,
    ):
        self.model = model
        self.num_retries = num_retries
        self._total_cost: float = 0.0
        self._total_tokens_in: int = 0
        self._total_tokens_out: int = 0
        self._cost_lock = threading.Lock()

        self.completion_kwargs: dict[str, Any] = {
            **({"temperature": temperature} if temperature is not None else {}),
            **({"max_tokens": max_tokens} if max_tokens is not None else {}),
            **kwargs,
        }

    @property
    def total_cost(self) -> float:
        """Cumulative USD cost of all calls made through this LM instance."""
        return self._total_cost

    @property
    def total_tokens_in(self) -> int:
        """Cumulative input (prompt) tokens across all calls."""
        return self._total_tokens_in

    @property
    def total_tokens_out(self) -> int:
        """Cumulative output (completion) tokens across all calls."""
        return self._total_tokens_out

    def _check_truncation(self, choices: list[Any]) -> None:
        if any(getattr(c, "finish_reason", None) == "length" for c in choices):
            max_tok = self.completion_kwargs.get("max_tokens") or self.completion_kwargs.get(
                "max_completion_tokens"
            )
            logger.warning(
                f"LM response was truncated (finish_reason='length', max_tokens={max_tok}). "
                "Consider increasing max_tokens for better results."
            )

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        import litellm

        if isinstance(prompt, str):
            messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        else:
            messages = prompt

        completion = litellm.completion(
            model=self.model,
            messages=messages,
            num_retries=self.num_retries,
            drop_params=True,
            **self.completion_kwargs,
        )

        # Non-streaming calls always return ModelResponse (not CustomStreamWrapper)
        self._check_truncation(completion.choices)  # type: ignore[union-attr]

        # Accumulate cost
        try:
            cost = litellm.completion_cost(completion_response=completion) or 0.0  # type: ignore[attr-defined]
        except Exception:
            cost = 0.0

        # Accumulate token usage
        usage = getattr(completion, "usage", None)
        tokens_in = (getattr(usage, "prompt_tokens", 0) or 0) if usage is not None else 0
        tokens_out = (getattr(usage, "completion_tokens", 0) or 0) if usage is not None else 0

        with self._cost_lock:
            self._total_cost += cost
            self._total_tokens_in += tokens_in
            self._total_tokens_out += tokens_out

        choice = completion.choices[0]
        msg = choice.message
        content = getattr(msg, "content", None)

        if content is None:
            # Some providers may expose reasoning/content differently
            content = getattr(msg, "reasoning_content", None)

        if content is None:
            # Tool-call-only / empty response case
            finish_reason = getattr(choice, "finish_reason", None)
            raise RuntimeError(
                f"LiteLLM returned no message content. finish_reason={finish_reason!r}, "
                f"response={completion.model_dump() if hasattr(completion, 'model_dump') else completion!r}"
            )

        return content

    def batch_complete(
        self, messages_list: list[list[dict[str, Any]]], max_workers: int = 10, **kwargs: Any
    ) -> list[str]:
        """Run multiple completions in parallel using ``litellm.batch_completion``.

        Args:
            messages_list: List of message lists, one per request.
            max_workers: Maximum concurrent requests.
            **kwargs: Extra keyword arguments forwarded to ``litellm.batch_completion``
                (e.g. ``timeout``, ``api_base``).  These override any matching keys
                set during ``__init__``.

        Returns:
            List of response strings, one per input.
        """
        import litellm

        merged = {**self.completion_kwargs, **kwargs}
        responses = litellm.batch_completion(
            model=self.model,
            messages=messages_list,
            max_workers=max_workers,
            num_retries=self.num_retries,
            drop_params=True,
            **merged,
        )

        batch_cost = 0.0
        batch_tokens_in = 0
        batch_tokens_out = 0
        results: list[str] = []
        for i, resp in enumerate(responses):
            self._check_truncation(resp.choices)

            if resp.choices[0].message.content is None:  # Failed
                results.append(self(messages_list[i]))
            else:
                results.append(resp.choices[0].message.content.strip())
                try:
                    batch_cost += litellm.completion_cost(completion_response=resp) or 0.0  # type: ignore[attr-defined]
                except Exception:
                    pass
                usage = getattr(resp, "usage", None)
                if usage is not None:
                    batch_tokens_in += getattr(usage, "prompt_tokens", 0) or 0
                    batch_tokens_out += getattr(usage, "completion_tokens", 0) or 0

        with self._cost_lock:
            self._total_cost += batch_cost
            self._total_tokens_in += batch_tokens_in
            self._total_tokens_out += batch_tokens_out

        return results

    def __repr__(self) -> str:
        params = [f"model={self.model!r}"]
        for k, v in self.completion_kwargs.items():
            params.append(f"{k}={v!r}")
        return f"LM({', '.join(params)})"


class DecisionModel(LanguageModel):
    def __init__(self, model: str, num_retries: int = 3, **kwargs: Any):
        super().__init__(model, num_retries=num_retries, **kwargs)

    def __call__(
        self,
        questions: dict[str, Any],
        context: str | dict[str, Any] | list[dict[str, Any]] = "",
    ) -> dict[str, Any]:
        import litellm

        response = litellm.decisions(
            model=self.model,
            state=context,
            questions=questions,
            num_retries=self.num_retries,
            **self.completion_kwargs,
        )

        try:
            cost = litellm.completion_cost(completion_response=response, model=self.model) or 0.0
        except Exception:
            cost = 0.0

        usage = getattr(response, "usage", None)
        tokens_in = (getattr(usage, "input_tokens", 0) or 0) if usage is not None else 0
        tokens_out = (getattr(usage, "output_tokens", 0) or 0) if usage is not None else 0

        with self._cost_lock:
            self._total_cost += cost
            self._total_tokens_in += tokens_in
            self._total_tokens_out += tokens_out

        return response.model_dump()["answers"]
