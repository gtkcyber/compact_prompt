# API reference

The complete, auto-generated reference for the public API. New here? Start with
the [Home page](index.md) and its examples first.

## The main entry point

::: compactprompt.CompactPrompt

::: compactprompt.CompactResult

## Shortcuts

::: compactprompt.compact

::: compactprompt.abbreviate

::: compactprompt.restore

## Strategies

### Hard prompt pruning

::: compactprompt.HardPromptCompressor

::: compactprompt.HardPromptResult

### N-gram abbreviation

::: compactprompt.NgramAbbreviator

::: compactprompt.Abbreviation

### Numeric quantization

::: compactprompt.quantize

::: compactprompt.quantize_uniform

::: compactprompt.quantize_kmeans

::: compactprompt.quantize_dataframe

::: compactprompt.QuantizedColumn

### Few-shot example selection

::: compactprompt.select_examples

::: compactprompt.SelectionResult

### Measuring fidelity

::: compactprompt.cosine_fidelity

::: compactprompt.FidelityResult

## Alternative pruning engines

::: compactprompt.LLMLinguaCompressor

::: compactprompt.CavemanCompressor

## Scoring internals

::: compactprompt.StaticSelfInformation

::: compactprompt.LocalLMScorer

::: compactprompt.count_tokens
