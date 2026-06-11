"""Self-information scoring for Hard Prompt Compression.

This implements the hybrid scoring described in CompactPrompt (Choi et al.,
2025, Sec. 3.1):

* **Static self-information** ``I_stat(t) = -log2 p(t)`` — global token rarity
  estimated from a corpus (Wikipedia / ShareGPT / arXiv in the paper).
* **Dynamic self-information** ``s_dyn(t | c) = -log2 P_model(t | c)`` — the
  context-conditional surprisal from a scorer LLM.
* **Fusion rule** (Eq. 2-3): let ``delta = |s_dyn - s_stat| / s_stat``.
  If ``delta <= 0.1`` use the mean of the two scores; otherwise trust the
  dynamic score, which better captures context sensitivity.

All heavy dependencies are imported lazily so this module imports cleanly on a
bare Python install. The dynamic scorer is *pluggable*: pass any callable, or
use the bundled offline :class:`LocalLMScorer`.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Callable, List, Optional, Sequence, Tuple

_WORD_RE = re.compile(r"\w+", re.UNICODE)

# A token surprisal: (surface_text, char_start, char_end, information_bits).
Surprisal = Tuple[str, int, int, float]

# A pluggable dynamic scorer is any callable: text -> list of Surprisal.
DynamicScorer = Callable[[str], List[Surprisal]]


# ---------------------------------------------------------------------------
# Static self-information
# ---------------------------------------------------------------------------
class StaticSelfInformation:
    """Estimate ``I_stat(t) = -log2 p(t)`` from token probabilities.

    Build it one of three ways:

    * :meth:`from_corpus` — count unigrams in a corpus you supply (the faithful
      reproduction of the paper's offline corpus statistics).
    * :meth:`from_wordfreq` — use the ``wordfreq`` package's global frequency
      table (install the ``freq`` extra). A good zero-setup default.
    * :meth:`from_text` — bootstrap from the text being compressed itself. Needs
      nothing installed; rare words in the prompt score as informative.
    """

    def __init__(self, prob: Callable[[str], float], floor: float = 1e-9):
        self._prob = prob
        self._floor = floor

    def score(self, token: str) -> float:
        """Return the static self-information of ``token`` in bits."""
        p = self._prob(token.lower())
        return -math.log2(max(p, self._floor))

    # -- constructors -------------------------------------------------------
    @classmethod
    def from_corpus(cls, texts: Sequence[str], smoothing: float = 1.0) -> "StaticSelfInformation":
        """Build counts from ``texts`` with add-``smoothing`` (Laplace) probs."""
        counts: Counter = Counter()
        for t in texts:
            counts.update(w.lower() for w in _WORD_RE.findall(t))
        total = sum(counts.values())
        vocab = len(counts)
        denom = total + smoothing * (vocab + 1)

        def prob(tok: str) -> float:
            return (counts.get(tok, 0) + smoothing) / denom if denom else 0.0

        return cls(prob)

    @classmethod
    def from_text(cls, text: str, smoothing: float = 1.0) -> "StaticSelfInformation":
        """Bootstrap static statistics from a single document (no deps)."""
        return cls.from_corpus([text], smoothing=smoothing)

    @classmethod
    def from_wordfreq(cls, lang: str = "en") -> "StaticSelfInformation":
        """Use the global ``wordfreq`` frequency table (needs the ``freq`` extra)."""
        try:
            from wordfreq import word_frequency
        except Exception as exc:  # pragma: no cover - exercised only without dep
            raise ImportError(
                "StaticSelfInformation.from_wordfreq() needs the 'wordfreq' "
                "package. Install it with: pip install 'compactprompt[freq]'"
            ) from exc

        def prob(tok: str) -> float:
            return word_frequency(tok, lang)

        return cls(prob)


def default_static(text: Optional[str] = None) -> StaticSelfInformation:
    """Best available static scorer with no required setup.

    Prefers ``wordfreq`` if installed, otherwise bootstraps from ``text`` (or
    returns a uniform scorer when no text is supplied).
    """
    try:
        return StaticSelfInformation.from_wordfreq()
    except ImportError:
        if text is not None:
            return StaticSelfInformation.from_text(text)
        return StaticSelfInformation(lambda _tok: 1e-6)


# ---------------------------------------------------------------------------
# Dynamic self-information (pluggable; bundled offline default)
# ---------------------------------------------------------------------------
class LocalLMScorer:
    """Offline dynamic self-information via a local Hugging Face causal LM.

    Computes per-token surprisal ``-log2 P_model(t | context)`` by running the
    model once over the text and reading the log-probability assigned to each
    actual next token. Subword pieces are merged back to whole words by summing
    their surprisals (so the score aligns with word/phrase pruning).

    Requires the ``dynamic`` extra (``torch`` + ``transformers``). Defaults to
    GPT-2, which is small and downloads quickly; pass any causal LM name.
    """

    def __init__(self, model_name: str = "gpt2", device: Optional[str] = None):
        self.model_name = model_name
        self._device = device
        self._tokenizer = None
        self._model = None
        self._torch = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise ImportError(
                "LocalLMScorer needs PyTorch + transformers. Install them with: "
                "pip install 'compactprompt[dynamic]'"
            ) from exc
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self._model.eval()
        if self._device:
            self._model.to(self._device)

    def __call__(self, text: str) -> List[Surprisal]:
        self._ensure_loaded()
        torch = self._torch
        tok = self._tokenizer
        enc = tok(text, return_offsets_mapping=True, return_tensors="pt")
        input_ids = enc["input_ids"]
        offsets = enc["offset_mapping"][0].tolist()
        if self._device:
            input_ids = input_ids.to(self._device)

        with torch.no_grad():
            logits = self._model(input_ids).logits[0]
        log_probs = torch.log_softmax(logits, dim=-1)

        ids = input_ids[0].tolist()
        # Surprisal of token i is read from the distribution at position i-1.
        # The first token has no left context; give it the model's unconditional
        # estimate at position 0 (it still emits a distribution).
        out: List[Surprisal] = []
        for i, (tid, (start, end)) in enumerate(zip(ids, offsets)):
            if start == end:  # special / empty token, skip
                continue
            pos = max(i - 1, 0)
            lp = log_probs[pos, tid].item()  # natural log
            bits = -lp / math.log(2)
            out.append((text[start:end], start, end, bits))
        return out


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------
def combine(s_stat: float, s_dyn: float, delta_threshold: float = 0.1) -> float:
    """Fuse static and dynamic self-information (CompactPrompt Eq. 2-3).

    Args:
        s_stat: Static self-information ``I_stat(t)``.
        s_dyn: Dynamic self-information ``s_dyn(t | c)``.
        delta_threshold: The relative-difference cutoff (paper default 0.1).

    Returns:
        The combined importance score in bits.
    """
    if s_stat <= 0:
        return s_dyn
    delta = abs(s_dyn - s_stat) / s_stat
    if delta <= delta_threshold:
        return 0.5 * (s_stat + s_dyn)
    return s_dyn


def aggregate_word_surprisals(
    surprisals: List[Surprisal], spans: List[Tuple[int, int]]
) -> List[float]:
    """Sum subword surprisals into the unit ``span`` that contains each piece.

    Args:
        surprisals: Per-subword surprisals from a dynamic scorer.
        spans: ``(start, end)`` char ranges for the target units (e.g. words).

    Returns:
        One summed surprisal per span, in span order. Spans with no overlapping
        subword get ``0.0``.
    """
    totals = [0.0] * len(spans)
    for _, start, _end, bits in surprisals:
        for j, (s, e) in enumerate(spans):
            if s <= start < e:
                totals[j] += bits
                break
    return totals
