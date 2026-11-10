# TopoSWE: Topological Code-Graph Localization, AST Context Folding, and Deterministic Verification Gates for Local Gemma 4 Developer Agents

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Competition](https://img.shields.io/badge/Kaggle-Gemma%204%20Developer%20Agent%20Paper%20Track-orange.svg)](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper)

Official repository and research artifacts for the Kaggle writeup: **[TopoSWE: Code-Graph Localization & Verification Gates for Gemma 4 SWE Agents](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/writeups/toposwe-code-graph-gemma-4-swe-agents)**.

---

## 🚀 Key Highlights

* **+19.3% Absolute Resolution Gain:** Elevates local `Gemma-4-31B-IT` resolution on SWE-bench Lite ($N=300$) from **29.0% to 48.3%**, and `Gemma-4-9B-IT` from **18.3% to 33.7%**.
* **74.8% Prompt Token Footprint Reduction:** Submodular AST Context Folding compresses raw repository multi-file prompts into tight $\le 2{,}400$-token slices without attention dilution.
* **91.5% Fewer Syntax & Scope Regressions:** Pre-execution Syntax & Semantic Verification Gate (`SSVG`) verifies AST parsing, unbound symbols, and public method arities deterministically in $<5\text{ ms}$.
* **Zero External Graph Database:** Written in pure, dependency-light Python (`ast`, `numpy`, `google-adk`), requiring no heavy background daemons or external vector stores.

---

## 📁 Repository Structure

```
.
├── KAGGLE_WRITEUP.md        # Complete Kaggle Research Paper Writeup (<= 3,000 words)
├── pyproject.toml           # Package metadata and dependencies
├── submission.zip           # Evaluated Kaggle ADK package
├── submission/              # Declarative Google ADK agent directory
│   ├── manifest.yaml        # ADK SequentialAgent + LoopAgent orchestrator schema
│   ├── tools/               # Deterministic YAML tool definitions
│   ├── subagents/           # Isolated subagent specifications
│   ├── prompts/             # Subagent role prompts
│   └── adapters/            # Rank-16 Topo-DPO LoRA configuration
├── src/toposwe/             # Core library
│   ├── graph.py             # AST dependency multigraph parser
│   ├── localizer.py         # Hybrid Seeded Personalized PageRank & Context Folding
│   ├── verifier.py          # Deterministic SSVG pre-execution gate
│   ├── agent.py             # Multi-agent episode orchestrator
│   └── trainer.py           # Topology-Aware Direct Preference Optimization (Topo-DPO)
├── experiments/             # Benchmark execution & evaluation scripts
│   ├── generate_assets.py   # Pure-Python benchmark runner & asset renderer
│   └── run_benchmarks.py    # Micro-benchmark testbed
├── figures/                 # Paper figures & Kaggle card thumbnail (560x280)
├── results/                 # Evaluation JSON metrics
└── tests/                   # Test suite for graph, localizer, gate, and DPO
```

---

## 🛠️ Quickstart

### 1. Installation

```bash
git clone https://github.com/Ritabanm/gemma-4-developer-agent-paper.git
cd gemma-4-developer-agent-paper
pip install -e .
```

### 2. Run Tests

```bash
python3 -c "from tests.test_toposwe import *; test_graph_and_ppr_localization(); test_syntax_semantic_gate_catches_errors(); test_topo_dpo_metrics(); print('All tests passed!')"
```

### 3. Run Benchmark & Generate Figures

```bash
PYTHONPATH=src:. python3 experiments/generate_assets.py
```

---

## 📊 Summary of SWE-bench Lite Results

| Model & Configuration | % Resolved ($\uparrow$) | Hit@3 ($\uparrow$) | Prompt Tokens ($\downarrow$) | Syntax Errors ($\downarrow$) | Latency ($\downarrow$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Gemma 4 9B** (Zero-Shot ReAct) | 18.3% | 44.7% | 21.4k | 24.8% | 194.0s |
| **Gemma 4 9B + TopoSWE** | **33.7%** | **76.3%** | **5.8k** | **2.9%** | **78.5s** |
| **Gemma 4 31B** (Zero-Shot Baseline) | 29.0% | 58.0% | 24.6k | 16.4% | 268.0s |
| **Gemma 4 31B + TopoSWE (PPR + Folding)** | 41.0% | 81.3% | 6.4k | 9.1% | 118.0s |
| **Gemma 4 31B + TopoSWE (+ SSVG Gate)** | 44.7% | 81.3% | 6.2k | 1.8% | 96.4s |
| **Gemma 4 31B + Full TopoSWE (+ Topo-DPO)**| **48.3%** | **84.0%** | **5.9k** | **1.4%** | **89.2s** |

---

## 📜 Citation & License

This research is released under the **Apache-2.0 License**.

```bibtex
@article{toposwe2026gemma4,
  title={TopoSWE: Topological Code-Graph Localization, AST Context Folding, and Deterministic Verification Gates for Local Gemma 4 Developer Agents},
  author={TopoSWE Research Team},
  journal={Google - The Gemma 4 Developer Agent Paper Track, Kaggle},
  year={2026}
}
```
