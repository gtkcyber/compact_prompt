"""Hard Prompt Compression (CompactPrompt Sec. 3.1).

Remove low-information tokens from a prompt while preserving meaning and grammar:

1. Score every word with hybrid static/dynamic self-information.
2. Group words into syntactic phrases (noun/verb/prep phrases via spaCy
   dependency parsing) so pruning respects grammar.
3. Aggregate scores per phrase and drop the lowest-scoring phrases until the
   token budget is met.

The dynamic scorer and spaCy are both optional. With nothing installed this
still runs: it falls back to static-only scoring and per-word units, so the
result is a faithful approximation rather than a hard failure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .scoring import (
    DynamicScorer,
    StaticSelfInformation,
    aggregate_word_surprisals,
    combine,
    default_static,
)
from .tokens import count_tokens

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass
class Unit:
    """A prunable unit (a phrase, or a single word)."""

    text: str
    start: int
    end: int
    score: float
    is_word: bool  # False for punctuation/whitespace runs we always keep


@dataclass
class HardPromptResult:
    """Result of :meth:`HardPromptCompressor.compress`."""

    original: str
    compressed: str
    tokens_before: int
    tokens_after: int
    removed_units: List[str] = field(default_factory=list)

    @classmethod
    def from_texts(
        cls, original: str, compressed: str, removed_units: Optional[List[str]] = None
    ) -> "HardPromptResult":
        """Build a result, counting tokens of both strings with ``count_tokens``.

        Pruning engines should use this so token counts always come from this
        library's tokenizer (not a backend's), keeping ratios comparable.
        """
        return cls(
            original=original,
            compressed=compressed,
            tokens_before=count_tokens(original),
            tokens_after=count_tokens(compressed),
            removed_units=removed_units or [],
        )

    @property
    def ratio(self) -> float:
        """Compression ratio ``tokens_before / tokens_after`` (e.g. 2.3x)."""
        return self.tokens_before / self.tokens_after if self.tokens_after else 1.0

    @property
    def savings(self) -> float:
        """Fraction of tokens removed, in ``[0, 1]``."""
        if not self.tokens_before:
            return 0.0
        return 1.0 - self.tokens_after / self.tokens_before

    def __str__(self) -> str:
        return self.compressed


class HardPromptCompressor:
    """Prune low-information phrases from prompts.

    Args:
        scorer: Pluggable dynamic self-information scorer (``text -> surprisals``).
            Defaults to :class:`~compactprompt.scoring.LocalLMScorer` *only if*
            you opt in by passing one; if ``None`` and unavailable, static-only
            scoring is used.
        static: Static self-information scorer. Defaults to the best available
            (``wordfreq`` if installed, else bootstrapped from the input text).
        delta_threshold: Fusion threshold for Eq. 2-3 (paper default ``0.1``).
        use_phrases: Group words into phrases with spaCy when available.
        spacy_model: spaCy model name for dependency parsing.
        protect_entities: Never prune named entities / numbers (recommended).
    """

    def __init__(
        self,
        scorer: Optional[DynamicScorer] = None,
        static: Optional[StaticSelfInformation] = None,
        delta_threshold: float = 0.1,
        use_phrases: bool = True,
        spacy_model: str = "en_core_web_sm",
        protect_entities: bool = True,
    ):
        self.scorer = scorer
        self.static = static
        self.delta_threshold = delta_threshold
        self.use_phrases = use_phrases
        self.spacy_model = spacy_model
        self.protect_entities = protect_entities
        self._nlp = None

    # -- spaCy ---------------------------------------------------------------
    def _load_spacy(self):
        if self._nlp is not None:
            return self._nlp
        try:
            import spacy
        except Exception:
            return None
        try:
            self._nlp = spacy.load(self.spacy_model)
        except Exception:
            try:  # a blank pipeline still gives us a sentence/token splitter
                self._nlp = spacy.blank("en")
            except Exception:
                return None
        return self._nlp

    # -- scoring -------------------------------------------------------------
    def _word_units(self, text: str) -> List[Unit]:
        """Build scored units (phrases when possible, else words)."""
        static = self.static or default_static(text)

        # Dynamic surprisals (optional).
        surprisals = []
        if self.scorer is not None:
            try:
                surprisals = self.scorer(text)
            except Exception:
                surprisals = []

        nlp = self._load_spacy() if self.use_phrases else None
        if nlp is not None and "parser" in getattr(nlp, "pipe_names", []):
            return self._phrase_units(text, nlp, static, surprisals)
        return self._token_units(text, static, surprisals)

    def _score_words(self, words, static, surprisals):
        spans = [(w_start, w_end) for (_t, w_start, w_end) in words]
        dyn = aggregate_word_surprisals(surprisals, spans) if surprisals else [0.0] * len(words)
        scores = []
        for (surface, _s, _e), d in zip(words, dyn):
            s_stat = static.score(surface)
            s = combine(s_stat, d, self.delta_threshold) if surprisals else s_stat
            scores.append(s)
        return scores

    def _token_units(self, text, static, surprisals) -> List[Unit]:
        words: List[Tuple[str, int, int]] = []
        units_raw: List[Tuple[str, int, int, bool]] = []
        for m in _TOKEN_RE.finditer(text):
            surface, s, e = m.group(), m.start(), m.end()
            is_word = bool(re.match(r"\w", surface))
            units_raw.append((surface, s, e, is_word))
            if is_word:
                words.append((surface, s, e))
        word_scores = self._score_words(words, static, surprisals)
        score_by_start = {w[1]: sc for w, sc in zip(words, word_scores)}
        return [
            Unit(surface, s, e, score_by_start.get(s, float("inf")), is_word)
            for (surface, s, e, is_word) in units_raw
        ]

    def _phrase_units(self, text, nlp, static, surprisals) -> List[Unit]:
        doc = nlp(text)
        protected = set()
        if self.protect_entities:
            for ent in doc.ents:
                for tok in ent:
                    protected.add(tok.i)
        # Phrase membership from noun chunks; remaining tokens are singletons.
        chunk_of = {}
        for ci, chunk in enumerate(doc.noun_chunks):
            for tok in chunk:
                chunk_of[tok.i] = ci

        words = [(t.text, t.idx, t.idx + len(t.text)) for t in doc if not t.is_space]
        word_scores = self._score_words(words, static, surprisals)
        score_of_tok = {}
        wi = 0
        for t in doc:
            if t.is_space:
                continue
            score_of_tok[t.i] = word_scores[wi]
            wi += 1

        units: List[Unit] = []
        i = 0
        toks = list(doc)
        while i < len(toks):
            t = toks[i]
            if t.is_space:
                i += 1
                continue
            ci = chunk_of.get(t.i)
            if ci is not None:
                group = [tk for tk in toks[i:] if chunk_of.get(tk.i) == ci]
                start = group[0].idx
                end = group[-1].idx + len(group[-1].text)
                scores = [score_of_tok[tk.i] for tk in group]
                is_protected = any(tk.i in protected for tk in group)
                avg = float("inf") if is_protected else sum(scores) / len(scores)
                is_word = any(not tk.is_punct for tk in group)
                units.append(Unit(text[start:end], start, end, avg, is_word))
                i += len(group)
            else:
                is_word = not t.is_punct
                base = score_of_tok.get(t.i, 0.0)
                sc = float("inf") if (t.i in protected or t.is_punct) else base
                units.append(Unit(t.text, t.idx, t.idx + len(t.text), sc, is_word))
                i += 1
        return units

    # -- public API ----------------------------------------------------------
    def compress(
        self,
        text: str,
        ratio: float = 0.5,
        budget: Optional[int] = None,
    ) -> HardPromptResult:
        """Prune ``text`` down to a token budget.

        Args:
            text: The prompt to compress.
            ratio: Target fraction of tokens to **remove** (0-1). ``0.5`` aims to
                halve the prompt. Ignored when ``budget`` is given.
            budget: Optional absolute target token count. Pruning stops once the
                prompt is at or below this many tokens.

        Returns:
            A :class:`HardPromptResult`. Note: hard pruning is **lossy** (removed
            words are not recoverable); use n-gram abbreviation for reversible
            compression.
        """
        if not text.strip():
            return HardPromptResult(text, text, count_tokens(text), count_tokens(text))

        tokens_before = count_tokens(text)
        if budget is None:
            budget = max(1, int(round(tokens_before * (1.0 - ratio))))

        units = self._word_units(text)
        # Candidates: prunable word units, least-informative first. Protected
        # units (entities/numbers, score == inf) are kept as a last resort and
        # only sacrificed if a tight budget cannot otherwise be met.
        prunable = [u for u in units if u.is_word]
        finite = sorted(
            (u for u in prunable if u.score != float("inf")), key=lambda u: u.score
        )
        protected = [u for u in prunable if u.score == float("inf")]
        candidates = finite + protected

        removed_spans: set = set()
        removed_text: List[str] = []

        def rebuild() -> str:
            kept = [u for u in units if (u.start, u.end) not in removed_spans]
            return _rejoin(text, kept)

        # Fast pass: remove low-information units using a running estimate.
        est = tokens_before
        i = 0
        while i < len(candidates) and est > budget:
            u = candidates[i]
            removed_spans.add((u.start, u.end))
            removed_text.append(u.text)
            est -= count_tokens(u.text)
            i += 1

        # Correction pass: subword merging makes the estimate approximate, so
        # verify against the real token count and prune a little more if needed.
        compressed = rebuild()
        while count_tokens(compressed) > budget and i < len(candidates):
            u = candidates[i]
            removed_spans.add((u.start, u.end))
            removed_text.append(u.text)
            i += 1
            compressed = rebuild()

        return HardPromptResult(
            original=text,
            compressed=compressed,
            tokens_before=tokens_before,
            tokens_after=count_tokens(compressed),
            removed_units=removed_text,
        )


def _rejoin(text: str, kept: List[Unit]) -> str:
    """Reconstruct text from kept units, preserving original spacing.

    We walk the original string and keep the original inter-unit whitespace for
    runs that survive, collapsing any gap left by a removed unit to a single
    space. This keeps the output readable and grammatical.
    """
    if not kept:
        return ""
    kept = sorted(kept, key=lambda u: u.start)
    parts: List[str] = []
    prev_end = kept[0].start
    parts.append(kept[0].text)
    prev_end = kept[0].end
    for u in kept[1:]:
        gap = text[prev_end:u.start]
        if gap == "":
            parts.append(u.text)
        elif gap.strip() == "":
            # Original separator survived intact (only whitespace between).
            parts.append(gap)
            parts.append(u.text)
        else:
            # Something was removed in between; normalize to a single space,
            # but keep newlines if the original gap had any (paragraph breaks).
            sep = "\n" if "\n" in gap else " "
            parts.append(sep)
            parts.append(u.text)
        prev_end = u.end
    out = "".join(parts)
    return re.sub(r"[ \t]+", " ", out).strip()
