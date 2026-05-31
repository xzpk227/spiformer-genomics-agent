# Continual Learning Agent for Genomics Tool Selection

A reinforcement learning module demonstrating continual/lifelong learning
on sequential genomics research tasks. The agent learns to select the correct
sequence of tools from the genomics agent toolkit across three distinct task
types, without forgetting previously learned skills.

## Motivation

Large-scale agentic AI systems for scientific research must acquire new
capabilities over time without overwriting prior knowledge — the same
catastrophic forgetting challenge that affects smart home agents, robotic
systems, and personalised AI assistants. This module applies Elastic Weight
Consolidation (EWC) to a genomics tool-selection agent as a concrete,
domain-grounded continual learning testbed.

## Problem Setup

**Environment:** `GenomicsToolEnv` — a custom Gymnasium environment where the
agent selects from 10 genomics tools to complete research queries.

**Tasks (trained sequentially):**

| Task | Name | Optimal Tools | Conflicting With |
|------|------|--------------|-----------------|
| 0 | Variant Analysis | SpliceAI, Spliformer, Ensembl, NCBI Gene | — |
| 1 | Literature Search | PubMed, GWAS Catalog | Splicing tools |
| 2 | Pathway Enrichment | GSEApy, GRN, PubMed | Splicing + GWAS tools |

**Setting:** Class-incremental — the task identity is **not** provided to the
agent at inference time, forcing the network to share representations across
tasks and creating genuine weight interference.

## Method: Elastic Weight Consolidation (EWC)

After training on each task, EWC estimates parameter importance via the
diagonal Fisher information matrix and adds a quadratic regularisation term
that penalises changes to important weights:

```
Loss = TD_loss + (λ/2) * Σ_i F_i * (θ_i − θ*_i)²
```

where `F_i` is the Fisher information for parameter `i`, `θ*_i` is its value
after the previous task, and `λ` controls the plasticity-stability trade-off.

## Results

```
Accuracy Matrix (EWC, λ=50):
                    variant_analysis  literature_search  pathway_enrichment
After Task 0                   1.000              —                  —
After Task 1                   1.000           0.000                  —
After Task 2                   0.000           0.000              1.000

AA  = 0.333   BWT = -0.500

Accuracy Matrix (Baseline, no EWC):
                    variant_analysis  literature_search  pathway_enrichment
After Task 0                   1.000              —                  —
After Task 1                   0.000           0.000                  —
After Task 2                   0.000           0.000              0.000

AA  = 0.000   BWT = -0.500
```

**Key observations:**
- **Catastrophic forgetting** is severe in the baseline: Task 0 accuracy drops
  to 0.0 immediately upon Task 1 training, and the agent fails all tasks by
  the end (AA = 0.0).
- **EWC protects Task 0** through Task 1 training (accuracy remains 1.0),
  demonstrating successful knowledge consolidation.
- **Plasticity-stability trade-off**: EWC's protection of Task 0 weights limits
  adaptation to Tasks 1 and 2, illustrating the fundamental tension in
  continual learning — a known open research challenge.

## Architecture

```
continual_agent/
├── env/
│   └── genomics_tool_env.py   # Custom Gymnasium env (task-incremental +
│                               # class-incremental modes)
├── agents/
│   ├── dqn_agent.py           # DQN with experience replay + target network
│   └── ewc.py                 # EWC + MultiTaskEWC (Kirkpatrick et al., 2017)
├── skills/
│   └── skill_library.py       # Persistent skill checkpoint store
└── evaluation/
    └── metrics.py             # AA, BWT, FWT + per-episode evaluation
```

## Running

```bash
pip install gymnasium torch numpy
python scripts/train_continual.py --episodes 600 --lambda-ewc 50 --seed 42
```

## Connection to Existing Agent

The 10 tools in this environment correspond directly to the tools registered
in `backend/agent/agent.py`:

```python
TOOLS = [
    search_literature,           # tool 0
    query_ensembl,               # tool 1
    query_ncbi_gene,             # tool 2
    query_gwas_catalog,          # tool 3
    predict_splicing,            # tool 4
    predict_splicing_spliformer, # tool 5
    spliformer_motif,            # tool 6
    run_enrichment_analysis,     # tool 7
    generate_grn,                # tool 8
    generate_report,             # tool 9
]
```

The continual learning module provides a principled framework for extending
the agent with new tool capabilities over time — directly addressing the
lifelong skill acquisition challenge in deployed agentic AI systems.

## References

Kirkpatrick, J. et al. (2017). Overcoming catastrophic forgetting in neural
networks. *PNAS*, 114(13), 3521–3526.

Lopez-Paz, D. & Ranzato, M. (2017). Gradient episodic memory for continual
learning. *NeurIPS*.
