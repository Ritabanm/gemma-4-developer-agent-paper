"""
TopoSWE: Topological Code-Graph Localization, Budgeted Subgraph Context Folding,
and Deterministic Verification Gates for Local Gemma 4 Developer Agents.
"""

from toposwe.graph import CodeEntity, CodeGraph, EdgeType, build_repository_graph
from toposwe.localizer import (
    BudgetedContextSlice,
    TopoLocalizer,
    compute_bm25_scores,
    personalized_pagerank,
)
from toposwe.verifier import (
    PatchVerificationResult,
    SearchReplaceBlock,
    SyntaxSemanticGate,
    apply_search_replace,
)
from toposwe.agent import (
    AgentTrajectoryStep,
    SubagentHandoffState,
    TopoSWEOrchestrator,
)
from toposwe.trainer import (
    PreferencePair,
    TopoDPOLossConfig,
    compute_topo_dpo_metrics,
    filter_trajectories_for_dpo,
)

__version__ = "0.1.0"
__all__ = [
    "CodeEntity",
    "CodeGraph",
    "EdgeType",
    "build_repository_graph",
    "BudgetedContextSlice",
    "TopoLocalizer",
    "compute_bm25_scores",
    "personalized_pagerank",
    "PatchVerificationResult",
    "SearchReplaceBlock",
    "SyntaxSemanticGate",
    "apply_search_replace",
    "AgentTrajectoryStep",
    "SubagentHandoffState",
    "TopoSWEOrchestrator",
    "PreferencePair",
    "TopoDPOLossConfig",
    "compute_topo_dpo_metrics",
    "filter_trajectories_for_dpo",
]

