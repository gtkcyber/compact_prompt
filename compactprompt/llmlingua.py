"""LLMLingua backend for hard-prompt compression.

`compactprompt`'s pruning engine is swappable: anything exposing
``compress(text, ratio=, budget=) -> HardPromptResult`` can be plugged into the
:class:`~compactprompt.pipeline.CompactPrompt` pipeline. This module wraps
Microsoft's `LLMLingua <https://github.com/microsoft/LLMLingua>`_ —
a mature, well-benchmarked perplexity-based prompt compressor — as exactly such
an engine, so you can use LLMLingua's token dropping in place of (or alongside)
the built-in self-information pruner.

Install the optional dependency::

    pip install 'compactprompt[llmlingua]'

Usage::

    from compactprompt import CompactPrompt
    from compactprompt.llmlingua import LLMLinguaCompressor

    pruner = LLMLinguaCompressor()                       # LLMLingua-2, CPU by default
    result = CompactPrompt.compact(prompt, pruner=pruner)
    # or the shortcut:
    result = CompactPrompt.compact(prompt, engine="llmlingua")
"""

from __future__ import annotations

from typing import Optional

from .hard_prompt import HardPromptResult

# A small, CPU-friendly LLMLingua-2 model. The LLMLingua v1 default
# (Llama-2-7B on CUDA) is far heavier and won't run on most machines.
DEFAULT_MODEL = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"


class LLMLinguaCompressor:
    """Adapter exposing LLMLingua as a `compactprompt` pruning engine.

    Args:
        model_name: Hugging Face model for LLMLingua. Defaults to a compact
            LLMLingua-2 model.
        use_llmlingua2: Use the LLMLingua-2 token-classification compressor
            (recommended; faster and matches ``DEFAULT_MODEL``).
        device_map: Torch device (``"cpu"``, ``"cuda"``, ...). Defaults to CPU.
        **compressor_kwargs: Forwarded to ``llmlingua.PromptCompressor``.

    The underlying model is loaded lazily on first :meth:`compress` call (or via
    :meth:`load`), so constructing this object is cheap and import-safe.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        use_llmlingua2: bool = True,
        device_map: str = "cpu",
        **compressor_kwargs,
    ):
        self.model_name = model_name
        self.use_llmlingua2 = use_llmlingua2
        self.device_map = device_map
        self._compressor_kwargs = compressor_kwargs
        self._compressor = None

    def load(self):
        """Eagerly load the LLMLingua model (otherwise loaded on first use)."""
        if self._compressor is not None:
            return self._compressor
        try:
            from llmlingua import PromptCompressor
        except Exception as exc:
            raise ImportError(
                "LLMLinguaCompressor needs the 'llmlingua' package. Install it "
                "with: pip install 'compactprompt[llmlingua]'"
            ) from exc
        self._compressor = PromptCompressor(
            model_name=self.model_name,
            use_llmlingua2=self.use_llmlingua2,
            device_map=self.device_map,
            **self._compressor_kwargs,
        )
        return self._compressor

    def compress(
        self,
        text: str,
        ratio: float = 0.5,
        budget: Optional[int] = None,
        instruction: str = "",
        question: str = "",
        **compress_kwargs,
    ) -> HardPromptResult:
        """Compress ``text`` with LLMLingua, returning a :class:`HardPromptResult`.

        Args:
            text: The prompt to compress.
            ratio: Target fraction of tokens to **remove** (0-1). Mapped to
                LLMLingua's ``rate`` (keep-fraction) as ``1 - ratio``. Ignored
                when ``budget`` is set.
            budget: Absolute target token count. Mapped to LLMLingua's
                ``target_token``.
            instruction / question: Optional LLMLingua query-aware fields
                (LLMLingua keeps tokens relevant to these).
            **compress_kwargs: Forwarded to ``PromptCompressor.compress_prompt``.

        Returns:
            A :class:`HardPromptResult`. Token counts use this library's
            :func:`~compactprompt.tokens.count_tokens` for consistency with the
            rest of the pipeline.
        """
        compressor = self.load()
        params = {"instruction": instruction, "question": question}
        if budget is not None:
            params["target_token"] = int(budget)
        else:
            params["rate"] = max(0.0, min(1.0, 1.0 - ratio))
        params.update(compress_kwargs)

        result = compressor.compress_prompt([text], **params)
        if isinstance(result, dict):
            compressed = result.get("compressed_prompt", "")
        else:  # some versions return the string directly
            compressed = str(result)

        return HardPromptResult.from_texts(text, compressed)
