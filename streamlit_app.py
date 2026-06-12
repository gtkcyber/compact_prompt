"""Tiny Streamlit demo for compactprompt.

Run with:
    pip install -e '.[app]'        # or: pip install streamlit
    streamlit run streamlit_app.py

Enter a prompt, pick how aggressively to compact it, and see the compressed
result and token savings side by side.
"""

import importlib
import shutil

import streamlit as st

from compactprompt import CompactPrompt, cosine_fidelity
from compactprompt.scoring import LocalLMScorer

EXAMPLE = (
    "Please could you very kindly go ahead and provide a really concise summary "
    "of the quarterly financial report that was prepared for the board of "
    "directors, focusing primarily on revenue and net income for this period."
)


def have(module: str) -> bool:
    """Return True if an optional dependency is importable."""
    try:
        importlib.import_module(module)
        return True
    except Exception:  # pragma: no cover - UI guard
        return False


@st.cache_resource(show_spinner="Loading scorer model (first run downloads gpt2)…")
def load_scorer(model_name: str) -> LocalLMScorer:
    """Cache the offline dynamic scorer across reruns."""
    instance = LocalLMScorer(model_name)
    instance("warm up")  # force the lazy model load now, inside the cached call
    return instance


@st.cache_resource(show_spinner="Loading LLMLingua model (first run downloads it)…")
def load_llmlingua():
    """Cache the LLMLingua pruning engine across reruns."""
    from compactprompt import LLMLinguaCompressor

    engine = LLMLinguaCompressor()
    engine.load()
    return engine


def sidebar() -> dict:
    """Render the settings sidebar and return the chosen options."""
    has_dynamic = have("torch") and have("transformers")
    has_phrases = have("spacy")
    has_embeddings = have("sentence_transformers")
    has_llmlingua = have("llmlingua")
    has_caveman = have("anthropic") or bool(shutil.which("claude"))

    with st.sidebar:
        st.header("Settings")

        opts = {}
        opts["prune"] = st.checkbox("Hard-prompt pruning (lossy)", value=True)
        engines = ["Built-in"]
        if has_llmlingua:
            engines.append("LLMLingua")
        if has_caveman:
            engines.append("Caveman")
        opts["engine"] = st.radio(
            "Pruning engine",
            engines,
            horizontal=True,
            disabled=not opts["prune"],
            help="Built-in self-information pruning; Microsoft LLMLingua "
                 "(perplexity-based); or Caveman (LLM rewrites prose tersely).",
        )
        is_rewrite = opts["engine"] in ("LLMLingua", "Caveman")
        is_caveman = opts["engine"] == "Caveman"
        remove_pct = st.slider(
            "Tokens to remove",
            min_value=0,
            max_value=90,
            value=50,
            step=5,
            format="%d%%",
            help="Target percentage of tokens to drop via pruning.",
            disabled=not opts["prune"] or is_caveman,
        )
        opts["ratio"] = remove_pct / 100.0
        if is_caveman:
            st.caption("🪓 Caveman rewrites prose to its own degree — the ratio is ignored.")
        opts["use_phrases"] = st.checkbox(
            "Grammar-preserving phrases (spaCy)",
            value=has_phrases,
            disabled=not has_phrases or is_rewrite,
            help="Prune whole phrases instead of single words. Built-in engine only.",
        )
        opts["use_dynamic"] = st.checkbox(
            "Context-aware scoring (gpt2)",
            value=False,
            disabled=not has_dynamic or is_rewrite,
            help="Score tokens with a local language model. Built-in engine only.",
        )

        st.divider()
        opts["abbreviate"] = st.checkbox(
            "Reversible n-gram abbreviation",
            value=False,
            help="Lossless: replaces frequent phrases with placeholders you can undo.",
        )
        opts["ngram"] = st.slider("N-gram length", 2, 5, 2, disabled=not opts["abbreviate"])
        opts["top_k"] = st.number_input(
            "Max patterns (top-K)", min_value=1, max_value=500, value=100,
            disabled=not opts["abbreviate"],
        )

        st.divider()
        opts["show_fidelity"] = st.checkbox(
            "Measure semantic fidelity",
            value=False,
            disabled=not has_embeddings,
            help="Cosine similarity of meaning before/after. Needs the 'embeddings' extra.",
        )

        if not has_dynamic:
            st.caption("ℹ️ Install `.[dynamic]` for context-aware scoring.")
        if not has_phrases:
            st.caption("ℹ️ Install `.[phrases]` + `spacy download en_core_web_sm`.")
        if not has_embeddings:
            st.caption("ℹ️ Install `.[embeddings]` to measure fidelity.")
        if not has_llmlingua:
            st.caption("ℹ️ Install `.[llmlingua]` to prune with LLMLingua.")
        if not has_caveman:
            st.caption("ℹ️ Install `.[caveman]` (or the `claude` CLI) for the Caveman engine.")
    return opts


def show_results(result, show_fidelity: bool) -> None:
    """Render metrics, the before/after columns, and optional fidelity."""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tokens before", result.tokens_before)
    c2.metric("Tokens after", result.tokens_after,
              delta=result.tokens_after - result.tokens_before)
    c3.metric("Compression", f"{result.ratio:.2f}×")
    c4.metric("Tokens saved", f"{result.savings:.0%}")

    left, right = st.columns(2)
    with left:
        st.subheader("Original")
        st.write(result.original)
    with right:
        st.subheader("Compressed")
        st.write(result.text)

    if result.dictionary:
        with st.expander(
            f"Abbreviation legend ({len(result.dictionary)} patterns) — reversible"
        ):
            st.json(result.dictionary)
            st.caption("`result.restore()` reproduces the text from this legend.")

    if show_fidelity:
        try:
            with st.spinner("Embedding for fidelity…"):
                fid = cosine_fidelity(result.original, result.text)
            st.metric("Semantic fidelity (cosine similarity)", f"{fid.mean:.3f}")
        except Exception as exc:  # pragma: no cover - UI guard
            st.warning(f"Could not compute fidelity: {exc}")


def main() -> None:
    """Render the demo. Streamlit re-runs this top to bottom on every interaction."""
    st.set_page_config(page_title="CompactPrompt", page_icon="🗜️", layout="wide")
    st.title("🗜️ CompactPrompt")
    st.caption(
        "Shrink prompts with the strategies from "
        "[CompactPrompt](https://arxiv.org/abs/2510.18043) (Choi et al., 2025)."
    )

    opts = sidebar()
    prompt = st.text_area("Prompt", value=EXAMPLE, height=160)
    if not st.button("Compact prompt", type="primary", disabled=not prompt.strip()):
        return

    engine = opts["engine"]
    is_rewrite = engine in ("LLMLingua", "Caveman")
    scorer = load_scorer("gpt2") if (opts["use_dynamic"] and not is_rewrite) else None
    pruner = None
    if engine == "LLMLingua":
        pruner = load_llmlingua()
    elif engine == "Caveman":
        from compactprompt import CavemanCompressor

        pruner = CavemanCompressor()
    spinner = f"Compressing with {engine}…" if is_rewrite else "Compressing…"
    try:
        with st.spinner(spinner):
            result = CompactPrompt.compact(
                prompt,
                ratio=opts["ratio"],
                prune=opts["prune"],
                abbreviate=opts["abbreviate"],
                ngram=opts["ngram"],
                top_k=int(opts["top_k"]),
                use_phrases=opts["use_phrases"],
                scorer=scorer,
                pruner=pruner,
            )
    except Exception as exc:  # pragma: no cover - surface config errors in UI
        st.error(f"Compaction failed: {exc}")
        return

    if result.text == result.original:
        st.warning(
            f"The {engine} engine returned the prompt unchanged — it may have "
            "had little to compress, or the LLM declined. Try a longer, more "
            "verbose prompt."
        )
    st.caption(f"Engine: **{engine}**")
    show_results(result, opts["show_fidelity"])


main()
