"""
Multi-Agent ADK Orchestrator for TopoSWE.

Implements Isolated Context-Bounded Subagent Handoff across three specialized roles:
  1. TopoLocalizerAgent: Builds repository AST graph and outputs a budgeted context slice
  2. PatchSynthesizerAgent: Generates minimal SEARCH/REPLACE patches conditioned only on the slice
  3. VerificationGateAgent: Applies deterministic SSVG checks and targeted test execution
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from toposwe.graph import CodeGraph, build_repository_graph
from toposwe.localizer import BudgetedContextSlice, TopoLocalizer
from toposwe.verifier import PatchVerificationResult, SearchReplaceBlock, SyntaxSemanticGate


@dataclass
class AgentTrajectoryStep:
    """Single step in a multi-agent episode trajectory."""

    step_id: int
    subagent_role: str
    action_type: str
    input_tokens: int
    output_tokens: int
    gate_passed: Optional[bool] = None
    summary: str = ""


@dataclass
class SubagentHandoffState:
    """Compact structured state passed between ADK subagents (prevents context bloat)."""

    issue_id: str
    issue_text: str
    context_slice: Optional[BudgetedContextSlice] = None
    candidate_edits: List[SearchReplaceBlock] = field(default_factory=list)
    verification_result: Optional[PatchVerificationResult] = None
    trajectory: List[AgentTrajectoryStep] = field(default_factory=list)
    resolved: bool = False
    syntax_rejections_caught: int = 0

    @property
    def total_context_tokens(self) -> int:
        return sum(s.input_tokens + s.output_tokens for s in self.trajectory)


PatchPolicyFn = Callable[[SubagentHandoffState, int], SearchReplaceBlock]


class TopoSWEOrchestrator:
    """
    Coordinates the 3-stage ADK subagent workflow with deterministic repair loops.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        token_budget: int = 2400,
        max_focal_entities: int = 5,
        max_repair_attempts: int = 3,
    ) -> None:
        self.localizer = TopoLocalizer(
            alpha=alpha,
            token_budget=token_budget,
            max_focal_entities=max_focal_entities,
        )
        self.gate = SyntaxSemanticGate(check_signature_compat=True)
        self.max_repair_attempts = max_repair_attempts

    def run_episode(
        self,
        repo_root: str,
        issue_id: str,
        issue_text: str,
        file_sources: Dict[str, str],
        patch_policy: PatchPolicyFn,
        test_oracle: Optional[Callable[[Dict[str, str]], bool]] = None,
        graph: Optional[CodeGraph] = None,
    ) -> SubagentHandoffState:
        """Execute a full localization -> synthesis -> verification episode."""
        if graph is None:
            graph = build_repository_graph(repo_root)

        state = SubagentHandoffState(issue_id=issue_id, issue_text=issue_text)

        # Stage 1: TopoLocalizerAgent
        ctx_slice = self.localizer.localize(graph, issue_text)
        state.context_slice = ctx_slice
        state.trajectory.append(
            AgentTrajectoryStep(
                step_id=1,
                subagent_role="TopoLocalizerAgent",
                action_type="localize_subgraph",
                input_tokens=max(64, len(issue_text) // 4),
                output_tokens=ctx_slice.total_tokens_used,
                gate_passed=True,
                summary=(
                    f"Localized {len(ctx_slice.focal_entities)} focal entities and "
                    f"{len(ctx_slice.neighbor_skeletons)} neighbor skeletons "
                    f"({ctx_slice.total_tokens_used}/{ctx_slice.raw_repo_tokens} tokens)."
                ),
            )
        )

        # Stage 2 & 3 Loop: PatchSynthesizerAgent <-> VerificationGateAgent
        for attempt in range(1, self.max_repair_attempts + 1):
            edit = patch_policy(state, attempt)
            state.candidate_edits.append(edit)

            feedback_tokens = (
                len(state.verification_result.format_feedback_prompt()) // 4
                if state.verification_result is not None
                else 0
            )
            synth_in_tokens = ctx_slice.total_tokens_used + feedback_tokens + 120
            synth_out_tokens = max(
                32, (len(edit.search_block) + len(edit.replace_block)) // 4
            )

            state.trajectory.append(
                AgentTrajectoryStep(
                    step_id=len(state.trajectory) + 1,
                    subagent_role="PatchSynthesizerAgent",
                    action_type=f"propose_patch_attempt_{attempt}",
                    input_tokens=synth_in_tokens,
                    output_tokens=synth_out_tokens,
                    summary=f"Proposed SEARCH/REPLACE patch on {edit.file_path}.",
                )
            )

            orig_src = file_sources.get(edit.file_path, "")
            ver_res = self.gate.verify_patch(orig_src, edit, graph=graph)
            state.verification_result = ver_res

            state.trajectory.append(
                AgentTrajectoryStep(
                    step_id=len(state.trajectory) + 1,
                    subagent_role="VerificationGateAgent",
                    action_type="static_syntax_semantic_gate",
                    input_tokens=synth_out_tokens,
                    output_tokens=len(ver_res.format_feedback_prompt()) // 4 + 12,
                    gate_passed=ver_res.passed,
                    summary=ver_res.format_feedback_prompt(),
                )
            )

            if not ver_res.passed:
                state.syntax_rejections_caught += 1
                continue

            # Deterministic gate passed -> run impacted unit test suite
            patched_files = dict(file_sources)
            patched_files[edit.file_path] = ver_res.patched_source
            tests_ok = test_oracle(patched_files) if test_oracle is not None else True
            if tests_ok:
                state.resolved = True
                break
            else:
                ver_res.passed = False
                ver_res.diagnostics.append(
                    f"Impacted test suite failed on {ver_res.impacted_tests or ['test_suite']}."
                )

        return state

