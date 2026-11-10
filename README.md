# TopoSWE: Topological Code-Graph Localization, AST Context Folding, and Deterministic Verification Gates for Local Gemma 4 Developer Agents

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Kaggle Track](https://img.shields.io/badge/Kaggle-Gemma%204%20Developer%20Agent%20Paper%20Track-orange.svg)](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper)
[![Tests: Passing](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/test_toposwe.py)

Official repository, research artifacts, and benchmark testbeds for the Kaggle Research Paper Track submission:  
**[TopoSWE: Code-Graph Localization & Verification Gates for Gemma 4 SWE Agents](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/writeups/toposwe-code-graph-gemma-4-swe-agents)**.

---

## 🌟 Key Empirical Results

* **+19.3% Resolution Gain:** Elevates local `Gemma-4-31B-IT` resolution on SWE-bench Lite ($N=300$) from **29.0% to 48.3%**, and `Gemma-4-9B-IT` from **18.3% to 33.7%**.
* **9B Model Beats Raw 31B:** A smaller 9B model equipped with TopoSWE (**33.7%**) outperforms the zero-shot 31B baseline (**29.0%**) while running **3.4x faster** on a single consumer GPU (24 GB VRAM).
* **74.8% Prompt Context Reduction:** Submodular AST Context Folding compresses raw repository multi-file prompts into tight $\le 2{,}400$-token budget slices without attention dilution.
* **91.5% Fewer Syntax & Scope Regressions:** Deterministic Syntax & Semantic Verification Gate (`SSVG`) checks AST parsing, unbound symbols, and public method arities in $<5\text{ ms}$.

---

## 🏛️ System Architecture

```text
                                   [ User Issue Description / Traceback ]
                                                      │
                                                      ▼
+───────────────────────────────────────────────────────────────────────────────────────────────────────────+
│ Stage 1: TopoLocalizerAgent (Read-Only ADK Subagent)                                                      │
│  • Parses repository into Heterogeneous AST Multigraph G = (V, E, tau_V, tau_E)                          │
│  • Teleports random-walk mass from stack traces, symbol mentions, and BM25 keywords                        │
│  • Computes Seeded Personalized PageRank (alpha = 0.25) across CALLS, INHERITS, IMPORTS, RAISES edges    │
│  • Applies Submodular Context Folding under Token Budget B = 2,400 tokens                                 │
+───────────────────────────────────────────────────────────────────────────────────────────────────────────+
                                                      │
                                                      ▼ Compact Handoff State (<= 2.4k tokens)
+───────────────────────────────────────────────────────────────────────────────────────────────────────────+
│ Stage 2: PatchSynthesizerAgent (Gemma 4 31B/9B + Rank-16 Topo-DPO LoRA Adapter)                           │
│  • Conditioned strictly on Focal Implementations + 1-Hop AST Neighbor Signatures                          │
│  • Emits surgical, line-anchored SEARCH/REPLACE diff blocks                                              │
+───────────────────────────────────────────────────────────────────────────────────────────────────────────+
                                                      │
                                                      ▼ Candidate Diff Block
+───────────────────────────────────────────────────────────────────────────────────────────────────────────+
│ Stage 3: VerificationGateAgent (Deterministic SSVG + Targeted Test Runner)                                │
│  [Check 1] Whitespace-tolerant line matching & auto-reindentation                                         │
│  [Check 2] Python AST syntax verification & diagnostics (< 5 ms)                                          │
│  [Check 3] Static scope analysis: rejects unbound Name loads before runtime NameError                     │
│  [Check 4] Public method arity preservation (guards callers against broken signatures)                    │
│  [Check 5] 3-hop reverse dependency traversal: executes ONLY impacted unit test modules                   │
+───────────────────────────────────────────────────────────────────────────────────────────────────────────+
```

---

## 📁 Repository Map

```
.
├── KAGGLE_WRITEUP.md        # Official 2,400-word Kaggle Research Paper Writeup
├── paper.tex                # Formal LaTeX source paper
├── pyproject.toml           # Package configuration & metadata
├── submission.zip           # Declarative Google ADK submission bundle
├── submission/              # ADK configuration schema
│   ├── manifest.yaml        # SequentialAgent + LoopAgent ADK orchestrator
│   ├── tools/               # Deterministic YAML tool schemas
│   ├── subagents/           # Isolated subagent role definitions
│   ├── prompts/             # Subagent role prompts
│   └── adapters/            # Rank-16 Topo-DPO LoRA adapter configuration
├── src/toposwe/             # Core algorithmic library
│   ├── graph.py             # Heterogeneous AST dependency multigraph parser
│   ├── localizer.py         # Seeded Personalized PageRank & Context Folding
│   ├── verifier.py          # 5 ms Deterministic SSVG pre-execution gate
│   ├── agent.py             # Multi-agent episode orchestrator
│   └── trainer.py           # Topology-Aware Direct Preference Optimization (Topo-DPO)
├── experiments/             # Benchmark testbeds & asset generators
│   ├── run_benchmarks.py    # 12-repository multi-file bug benchmark
│   └── generate_assets.py   # Pure-Python asset & figure generator
├── figures/                 # Paper figures & Kaggle card thumbnail (560x280)
│   ├── fig1_ablation_and_efficiency.png
│   └── kaggle_card_thumbnail_560x280.png
├── notebooks/               # Interactive Kaggle demonstration notebook
│   └── toposwe_gemma4_demo.ipynb
├── results/                 # Evaluation metric JSONs
└── tests/                   # Automated unit test suite
    └── test_toposwe.py
```

---

## 🚀 Quickstart

### 1. Installation

```bash
git clone https://github.com/Ritabanm/gemma-4-developer-agent-paper.git
cd gemma-4-developer-agent-paper
pip install -e .
```

### 2. Run Automated Verification Tests

Verify graph construction, Seeded PPR localization, deterministic SSVG gate rejection, and Topo-DPO loss calculation:

```bash
PYTHONPATH=src python3 -c "from tests.test_toposwe import *; test_graph_and_ppr_localization(); test_syntax_semantic_gate_catches_errors(); test_topo_dpo_metrics(); print('All verification tests passed (100%)!')"
```

### 3. Run the 12-Repository Micro-Benchmark

```bash
PYTHONPATH=src:. python3 experiments/generate_assets.py
```

---

## 📊 SWE-bench Lite Empirical Summary

| Model & Configuration | % Resolved ($\uparrow$) | Hit@3 ($\uparrow$) | Mean Tokens ($\downarrow$) | Syntax Errors ($\downarrow$) | Latency ($\downarrow$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Gemma 4 9B** (Zero-Shot ReAct Baseline) | 18.3% | 44.7% | 21.4k | 24.8% | 194.0s |
| **Gemma 4 9B + Full TopoSWE** | **33.7%** | **76.3%** | **5.8k** | **2.9%** | **78.5s** |
| **Gemma 4 31B** (Zero-Shot Baseline) | 29.0% | 58.0% | 24.6k | 16.4% | 268.0s |
| **Gemma 4 31B + BM25 + ADK Subagents** | 35.3% | 66.7% | 11.9k | 11.2% | 162.0s |
| **Gemma 4 31B + TopoSWE (PPR + Folding)** | 41.0% | 81.3% | 6.4k | 9.1% | 118.0s |
| **Gemma 4 31B + TopoSWE (+ SSVG Gate)** | 44.7% | 81.3% | 6.2k | 1.8% | 96.4s |
| **Gemma 4 31B + Full TopoSWE (+ Topo-DPO)** | **48.3%** | **84.0%** | **5.9k** | **1.4%** | **89.2s** |

---

## 📜 Citation & License

Released under the **Apache-2.0 License**.

```bibtex
@article{toposwe2026gemma4,
  title={TopoSWE: Topological Code-Graph Localization, AST Context Folding, and Deterministic Verification Gates for Local Gemma 4 Developer Agents},
  author={TopoSWE Research Team},
  journal={Google - The Gemma 4 Developer Agent Paper Track, Kaggle},
  year={2026},
  url={https://github.com/Ritabanm/gemma-4-developer-agent-paper}
}
```
