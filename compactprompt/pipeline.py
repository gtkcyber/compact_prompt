"""The unified ``CompactPrompt`` pipeline.

This is the easy front door to every strategy in the paper. The headline call
is::

    from compactprompt import CompactPrompt

    result = CompactPrompt.compact(my_prompt)
    print(result.text)        # the compressed prompt
    print(result.ratio)       # e.g. 2.1 (x fewer tokens)
    original = result.restore()   # undo the (lossless) n-gram step

``compact`` combines the two text-level strategies — reversible **n-gram
abbreviation** and lossy **hard-prompt pruning** — and every knob is optional
and documented. The data-level strategies (numeric quantization, few-shot
exemplar selection) are exposed as their own methods/functions because they
operate on tables and example sets rather than a single prompt string.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .hard_prompt import HardPromptCompressor
from .ngram import Abbreviation, NgramAbbreviator
from .scoring import DynamicScorer, StaticSelfInformation
from .tokens import count_tokens


@dataclass
class CompactResult:
    """The output of :meth:`CompactPrompt.compact`.

    Attributes:
        text: The final compressed prompt.
        original: The input prompt.
        tokens_before / tokens_after: Token counts.
        dictionary: Reversible n-gram mapping (empty if abbreviation was off).
        steps: Names of the strategies applied, in order.
        stats: Per-step diagnostic numbers.
    """

    text: str
    original: str
    tokens_before: int
    tokens_after: int
    dictionary: Dict[str, str] = field(default_factory=dict)
    steps: List[str] = field(default_factory=list)
    stats: Dict[str, object] = field(default_factory=dict)

    @property
    def ratio(self) -> float:
        """Compression ratio ``tokens_before / tokens_after`` (e.g. 2.3x)."""
        return self.tokens_before / self.tokens_after if self.tokens_after else 1.0

    @property
    def savings(self) -> float:
        """Fraction of tokens saved, in ``[0, 1]``."""
        if not self.tokens_before:
            return 0.0
        return 1.0 - self.tokens_after / self.tokens_before

    def restore(self) -> str:
        """Undo the lossless n-gram abbreviation step.

        Note: hard-prompt pruning is lossy, so this recovers the *pruned*
        prompt with abbreviations expanded, not the verbatim original.
        """
        return NgramAbbreviator.decompress(self.text, self.dictionary)

    def __str__(self) -> str:
        return self.text


class CompactPrompt:
    """Configurable facade over all CompactPrompt strategies.

    You can use the one-shot classmethod :meth:`compact` for the common case, or
    construct an instance to reuse an expensive scorer/embedder across calls::

        cp = CompactPrompt(scorer=my_llm_scorer)
        cp.compact(prompt_a)
        cp.compact(prompt_b)

    Args:
        scorer: Pluggable dynamic self-information scorer (``text -> surprisals``)
            for hard pruning. ``None`` uses static-only scoring unless you pass
            one. See :class:`~compactprompt.scoring.LocalLMScorer` for an offline
            option.
        static: Static self-information scorer. Defaults to best-available.
        delta_threshold: Static/dynamic fusion threshold (paper default 0.1).
        use_phrases: Group words into grammatical phrases (needs spaCy).
        spacy_model: spaCy model name.
        ngram: N-gram length for abbreviation (paper best: 2).
        top_k: Number of frequent n-grams to abbreviate.
        pruner: Custom pruning engine exposing
            ``compress(text, ratio=, budget=) -> HardPromptResult``. Defaults to
            the built-in :class:`~compactprompt.hard_prompt.HardPromptCompressor`.
            Pass a :class:`~compactprompt.llmlingua.LLMLinguaCompressor` to prune
            with LLMLingua instead. When supplied, the scorer/static/phrase
            options above are ignored (they configure the built-in engine).
    """

    def __init__(
        self,
        scorer: Optional[DynamicScorer] = None,
        static: Optional[StaticSelfInformation] = None,
        delta_threshold: float = 0.1,
        use_phrases: bool = True,
        spacy_model: str = "en_core_web_sm",
        ngram: int = 2,
        top_k: int = 100,
        pruner=None,
    ):
        self.hard = pruner or HardPromptCompressor(
            scorer=scorer,
            static=static,
            delta_threshold=delta_threshold,
            use_phrases=use_phrases,
            spacy_model=spacy_model,
        )
        self.ngram = ngram
        self.top_k = top_k

    # -- instance API --------------------------------------------------------
    def run(
        self,
        prompt: str,
        ratio: float = 0.5,
        budget: Optional[int] = None,
        prune: bool = True,
        abbreviate: bool = False,
        ngram: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> CompactResult:
        """Compress ``prompt`` with the configured strategies.

        See :meth:`compact` for argument documentation; this is the instance
        method it delegates to.
        """
        original = prompt
        tokens_before = count_tokens(prompt)
        text = prompt
        steps: List[str] = []
        stats: Dict[str, object] = {}

        if prune:
            res = self.hard.compress(text, ratio=ratio, budget=budget)
            text = res.compressed
            steps.append("hard_prompt")
            stats["hard_prompt"] = {
                "tokens_before": res.tokens_before,
                "tokens_after": res.tokens_after,
                "removed": len(res.removed_units),
            }

        dictionary: Dict[str, str] = {}
        if abbreviate:
            abbr: Abbreviation = NgramAbbreviator(
                n=ngram if ngram is not None else self.ngram,
                top_k=top_k if top_k is not None else self.top_k,
            ).compress(text)
            text = abbr.text
            dictionary = abbr.dictionary
            steps.append("ngram_abbreviation")
            stats["ngram_abbreviation"] = {"patterns": len(dictionary)}

        return CompactResult(
            text=text,
            original=original,
            tokens_before=tokens_before,
            tokens_after=count_tokens(text),
            dictionary=dictionary,
            steps=steps,
            stats=stats,
        )

    # -- one-shot classmethod (the headline API) -----------------------------
    @classmethod
    def compact(
        cls,
        prompt: str,
        *,
        ratio: float = 0.5,
        budget: Optional[int] = None,
        prune: bool = True,
        abbreviate: bool = False,
        ngram: int = 2,
        top_k: int = 100,
        scorer: Optional[DynamicScorer] = None,
        static: Optional[StaticSelfInformation] = None,
        delta_threshold: float = 0.1,
        use_phrases: bool = True,
        spacy_model: str = "en_core_web_sm",
        engine: str = "builtin",
        pruner=None,
    ) -> CompactResult:
        """Compress a prompt. **The main entry point.**

        Args:
            prompt: The prompt text to compress.
            ratio: Target fraction of tokens to **remove** via hard pruning
                (0-1). ``0.5`` aims to halve the prompt. Ignored if ``budget``
                is set.
            budget: Optional absolute target token count for hard pruning.
            prune: Apply lossy hard-prompt pruning (low-information phrases).
                On by default — the result is usable as-is.
            abbreviate: Also apply lossless, reversible n-gram abbreviation.
                Off by default: abbreviated text needs its ``dictionary`` as a
                legend to be interpretable downstream, so enable it when you
                control both ends (e.g. compressing attached documents).
            ngram: N-gram length for abbreviation (paper's best: 2).
            top_k: Number of frequent n-grams to abbreviate.
            scorer: Pluggable dynamic self-information scorer for context-aware
                pruning. ``None`` falls back to static-only scoring. Pass a
                :class:`~compactprompt.scoring.LocalLMScorer` for offline
                context-aware scoring, or any ``text -> surprisals`` callable.
            static: Static self-information scorer (defaults to best-available).
            delta_threshold: Static/dynamic fusion threshold (paper default 0.1).
            use_phrases: Preserve grammar by pruning whole phrases (needs spaCy).
            spacy_model: spaCy model name for phrase parsing.
            engine: Pruning engine. ``"builtin"`` (default) uses this library's
                self-information pruner; ``"llmlingua"`` uses LLMLingua with
                default settings (needs the ``llmlingua`` extra); ``"caveman"``
                uses LLM-based caveman-style compression (needs an LLM — see
                :class:`~compactprompt.caveman.CavemanCompressor`).
            pruner: An explicit pruning engine instance (overrides ``engine``).
                See :class:`~compactprompt.llmlingua.LLMLinguaCompressor`.

        Returns:
            A :class:`CompactResult`. ``result.text`` is the compressed prompt;
            ``result.restore()`` reverses the abbreviation step.

        Example:
            >>> from compactprompt import CompactPrompt
            >>> r = CompactPrompt.compact("Please kindly summarize ...")
            >>> r.text            # doctest: +SKIP
            >>> r.ratio           # doctest: +SKIP
        """
        if pruner is None and engine == "llmlingua":
            from .llmlingua import LLMLinguaCompressor

            pruner = LLMLinguaCompressor()
        elif pruner is None and engine == "caveman":
            from .caveman import CavemanCompressor

            pruner = CavemanCompressor()
        elif engine not in ("builtin", "llmlingua", "caveman"):
            raise ValueError("engine must be 'builtin', 'llmlingua', or 'caveman'")

        compactor = cls(
            scorer=scorer,
            static=static,
            delta_threshold=delta_threshold,
            use_phrases=use_phrases,
            spacy_model=spacy_model,
            ngram=ngram,
            top_k=top_k,
            pruner=pruner,
        )
        return compactor.run(
            prompt,
            ratio=ratio,
            budget=budget,
            prune=prune,
            abbreviate=abbreviate,
        )
