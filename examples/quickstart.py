"""Runnable tour of every compactprompt strategy.

Run with:  python examples/quickstart.py

Only the first two sections need zero dependencies. The quantization, exemplar
selection, and fidelity sections degrade gracefully if their optional extras
are not installed.
"""

import compactprompt as cp
from compactprompt import CompactPrompt

PROMPT = (
    "Please could you very kindly go ahead and provide a really concise summary "
    "of the quarterly financial report that was prepared for the board of "
    "directors, focusing primarily on revenue and net income."
)


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    section("1. Hard Prompt Compression (the headline API)")
    r = CompactPrompt.compact(PROMPT, ratio=0.4)
    print(f"{r.tokens_before} -> {r.tokens_after} tokens ({r.ratio:.2f}x, "
          f"{r.savings:.0%} saved)")
    print("compressed:", r.text)

    section("2. Reversible N-gram Abbreviation (lossless)")
    doc = ("operating cash flow rose. operating cash flow fell. "
           "operating cash flow held steady.")
    abbr = cp.abbreviate(doc, n=3)
    print("compressed:", abbr.text)
    print("dictionary:", abbr.dictionary)
    print("round-trip exact:", abbr.restore() == doc)

    section("3. Numerical Quantization (bounded error)")
    values = [1.0, 2.5, 3.3, 4.8, 9.2, 10.0]
    q = cp.quantize(values, method="uniform", bits=8)
    print("original     :", values)
    print("reconstructed:", [round(x, 3) for x in q.reconstruct()])
    print("max error    :", round(q.max_error, 5))

    section("4. Representative Example Selection (few-shot)")
    try:
        texts = [f"topic {i % 5}: example sentence number {i}" for i in range(40)]
        sel = cp.select_examples(texts, k_range=(3, 8))
        print(f"selected k*={sel.k_star} exemplars (silhouette={sel.silhouette:.3f})")
        for ex in sel.examples[:5]:
            print("  -", ex)
    except ImportError as exc:
        print("skipped (install extras):", exc)

    section("5. Semantic fidelity of the compression")
    try:
        f = cp.cosine_fidelity(PROMPT, r.text)
        print(f"cosine similarity mean={f.mean:.3f}, 5th pct={f.p5:.3f}")
    except ImportError as exc:
        print("skipped (install extras):", exc)


if __name__ == "__main__":
    main()
