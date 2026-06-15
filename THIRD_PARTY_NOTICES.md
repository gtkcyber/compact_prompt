# Third-Party Notices

`compactprompt` includes, adapts, or builds upon third-party work. This file
records the required attributions and license notices.

---

## 1. Caveman — ported source code

`compactprompt/caveman.py`, `compactprompt/markdown.py`, and
`compactprompt/files.py` contain **ports and adaptations** of the
`caveman-compress` skill from the **Caveman** project. The structure-preservation
validators (fenced-code / inline-code / URL / heading extraction), the
frontmatter and outer-fence handling, the LLM compression/fix prompts, the
validate-and-fix-retry flow, the natural-language file detection
(`detect.py`), and the sensitive-path refusal (`is_sensitive_path`) are derived
from Caveman's source. The code was adapted to operate on strings, to use a
pluggable LLM callable, and to conform to this library's pruning-engine and
file-layer interfaces.

- Project: Caveman
- Author: Julius Brussee
- Source: https://github.com/JuliusBrussee/caveman
- License: MIT

```
MIT License

Copyright (c) 2026 Julius Brussee

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 2. CompactPrompt — methodology

This library is an independent implementation of the methodology described in
the paper below. No source code from the authors is included; the algorithms
were reimplemented from the paper.

- Paper: *CompactPrompt: A Unified Pipeline for Prompt and Data Compression in
  LLM Workflows*
- Authors: Joong Ho Choi, Jiayang Zhao, Jeel Shah, Ritvika Sonawane, Vedant
  Singh, Avani Appalla, Will Flanagan, Filipe Condessa
- Reference: arXiv:2510.18043 — https://arxiv.org/abs/2510.18043

---

## 3. Optional runtime dependencies

The following projects are used as **optional dependencies** (imported at
runtime when their extras are installed); none of their source is vendored here.
Each is distributed under its own license, reproduced with its distribution:

| Project | Used for | License |
|---|---|---|
| LLMLingua | Alternative pruning engine (`engine="llmlingua"`) | MIT |
| Anthropic SDK | Default LLM caller for the Caveman engine | MIT |
| spaCy | Phrase-level dependency parsing | MIT |
| scikit-learn | K-means / silhouette for quantization & exemplar selection | BSD-3-Clause |
| sentence-transformers | `all-mpnet-base-v2` embeddings | Apache-2.0 |
| transformers, torch | Local LM scorer / model backends | Apache-2.0 / BSD-3-Clause |
| tiktoken | Token counting | MIT |
| wordfreq | Static self-information frequencies | MIT (data: mixed) |
| streamlit | Demo app | Apache-2.0 |

These attributions are provided in good faith; consult each project's own
distribution for the authoritative license text.
