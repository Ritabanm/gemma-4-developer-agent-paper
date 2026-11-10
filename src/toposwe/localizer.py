"""
Topological Fault Localization via Hybrid Seeded Personalized PageRank (PPR)
and Submodular Token-Budgeted Context Folding.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from toposwe.graph import CodeEntity, CodeGraph


_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL_SPLIT_RE = re.compile(r"(?<!^)(?=[A-Z])")
_STACK_FRAME_RE = re.compile(
    r'File ["\']?([^"\'\s,]+?\.py)["\']?,\s*line\s*(\d+)(?:,\s*in\s*([A-Za-z_][A-Za-z0-9_]*))?'
)


def tokenize_code_text(text: str) -> List[str]:
    """Split snake_case and CamelCase identifiers into normalized subtokens."""
    tokens: List[str] = []
    for raw in _IDENTIFIER_RE.findall(text):
        tokens.append(raw.lower())
        for part in raw.split("_"):
            if part:
                tokens.append(part.lower())
                for sub in _CAMEL_SPLIT_RE.split(part):
                    if sub:
                        tokens.append(sub.lower())
    return tokens


def compute_bm25_scores(
    graph: CodeGraph,
    issue_text: str,
    k1: float = 1.5,
    b: float = 0.75,
) -> Dict[str, float]:
    """Compute BM25 lexical relevance over code entities for an issue description."""
    entity_ids = graph.entity_ids()
    if not entity_ids:
        return {}

    query_terms = tokenize_code_text(issue_text)
    if not query_terms:
        return {eid: 0.0 for eid in entity_ids}

    docs: Dict[str, List[str]] = {}
    df: Dict[str, int] = {}
    total_len = 0

    for eid in entity_ids:
        ent = graph.entities[eid]
        doc_text = f"{ent.file_path} {ent.qualified_name} {ent.signature} {ent.docstring} {ent.source_code}"
        toks = tokenize_code_text(doc_text)
        docs[eid] = toks
        total_len += len(toks)
        for t in set(toks):
            df[t] = df.get(t, 0) + 1

    n_docs = len(entity_ids)
    avgdl = max(1.0, total_len / n_docs)
    scores: Dict[str, float] = {}

    for eid in entity_ids:
        toks = docs[eid]
        dl = len(toks)
        tf_map: Dict[str, int] = {}
        for t in toks:
            tf_map[t] = tf_map.get(t, 0) + 1

        score = 0.0
        for q in query_terms:
            tf = tf_map.get(q, 0)
            if tf == 0:
                continue
            n_q = df.get(q, 0)
            idf = math.log(1.0 + (n_docs - n_q + 0.5) / (n_q + 0.5))
            denom = tf + k1 * (1.0 - b + b * (dl / avgdl))
            score += idf * (tf * (k1 + 1.0)) / denom

        # Downweight coarse module containers relative to fine-grained functions/methods
        if graph.entities[eid].kind == "module":
            score *= 0.45
        scores[eid] = score

    return scores


def extract_seed_distribution(
    graph: CodeGraph,
    issue_text: str,
    bm25_weight: float = 0.45,
    symbol_weight: float = 0.30,
    trace_weight: float = 0.25,
) -> Dict[str, float]:
    """
    Construct the Personalized PageRank teleportation vector s in Delta^{|V|-1}
    by combining BM25 lexical similarity, explicit symbol mentions, and stack-trace frames.
    """
    entity_ids = graph.entity_ids()
    if not entity_ids:
        return {}

    bm25 = compute_bm25_scores(graph, issue_text)
    max_bm25 = max(bm25.values(), default=0.0)
    norm_bm25 = {
        eid: (val / max_bm25 if max_bm25 > 0 else 0.0) for eid, val in bm25.items()
    }

    # Explicit identifier matches in issue text
    raw_identifiers = set(_IDENTIFIER_RE.findall(issue_text))
    sym_scores: Dict[str, float] = {eid: 0.0 for eid in entity_ids}
    for eid in entity_ids:
        ent = graph.entities[eid]
        if ent.kind == "module":
            continue
        if ent.name in raw_identifiers or ent.qualified_name in issue_text:
            sym_scores[eid] += 1.0
        if ent.file_path in issue_text or ent.file_path.split("/")[-1] in issue_text:
            sym_scores[eid] += 0.5
    max_sym = max(sym_scores.values(), default=0.0)
    norm_sym = {
        eid: (val / max_sym if max_sym > 0 else 0.0) for eid, val in sym_scores.items()
    }

    # Stack trace frame extraction
    trace_scores: Dict[str, float] = {eid: 0.0 for eid in entity_ids}
    for match in _STACK_FRAME_RE.finditer(issue_text):
        fpath, lineno_str, func_name = match.group(1), match.group(2), match.group(3)
        lineno = int(lineno_str)
        for eid in entity_ids:
            ent = graph.entities[eid]
            if ent.kind == "module":
                continue
            if ent.file_path.endswith(fpath) or fpath.endswith(ent.file_path):
                if ent.start_line <= lineno <= ent.end_line:
                    trace_scores[eid] += 2.0
                elif func_name and ent.name == func_name:
                    trace_scores[eid] += 1.5
    max_trace = max(trace_scores.values(), default=0.0)
    norm_trace = {
        eid: (val / max_trace if max_trace > 0 else 0.0)
        for eid, val in trace_scores.items()
    }

    combined: Dict[str, float] = {}
    for eid in entity_ids:
        combined[eid] = (
            bm25_weight * norm_bm25[eid]
            + symbol_weight * norm_sym[eid]
            + trace_weight * norm_trace[eid]
        )

    total = sum(combined.values())
    if total <= 1e-12:
        uniform = 1.0 / len(entity_ids)
        return {eid: uniform for eid in entity_ids}
    return {eid: val / total for eid, val in combined.items()}


def personalized_pagerank(
    graph: CodeGraph,
    seed_dist: Dict[str, float],
    alpha: float = 0.25,
    max_iter: int = 60,
    tol: float = 1e-7,
) -> Dict[str, float]:
    """
    Compute stationary distribution pi of Seeded Personalized PageRank:
        pi^{(t+1)} = alpha * s + (1 - alpha) * P^T pi^{(t)}
    where P_{uv} is the edge-type-weighted transition matrix with dangling-node teleportation.
    """
    entity_ids = graph.entity_ids()
    n = len(entity_ids)
    if n == 0:
        return {}

    idx_map = {eid: i for i, eid in enumerate(entity_ids)}
    s = np.array([seed_dist.get(eid, 0.0) for eid in entity_ids], dtype=np.float64)
    s_sum = s.sum()
    if s_sum <= 1e-12:
        s = np.full(n, 1.0 / n, dtype=np.float64)
    else:
        s = s / s_sum

    # Build column-stochastic transition representation
    out_weights = np.zeros(n, dtype=np.float64)
    transitions: List[Tuple[int, int, float]] = []
    for edge in graph.edges:
        u = idx_map[edge.source]
        v = idx_map[edge.target]
        transitions.append((u, v, edge.weight))
        out_weights[u] += edge.weight

    pi = s.copy()
    for _ in range(max_iter):
        next_pi = alpha * s
        # Dangling mass redistributes according to personalization vector s
        dangling_mass = np.sum(pi[out_weights <= 1e-12])
        next_pi += (1.0 - alpha) * dangling_mass * s

        for u, v, w in transitions:
            if out_weights[u] > 1e-12:
                next_pi[v] += (1.0 - alpha) * pi[u] * (w / out_weights[u])

        if np.linalg.norm(next_pi - pi, ord=1) < tol:
            pi = next_pi
            break
        pi = next_pi

    return {eid: float(pi[idx_map[eid]]) for eid in entity_ids}


@dataclass
class BudgetedContextSlice:
    """Output of Token-Budgeted Subgraph Context Folding."""

    focal_entities: List[CodeEntity] = field(default_factory=list)
    neighbor_skeletons: List[CodeEntity] = field(default_factory=list)
    ranked_entity_ids: List[str] = field(default_factory=list)
    ppr_scores: Dict[str, float] = field(default_factory=dict)
    total_tokens_used: int = 0
    raw_repo_tokens: int = 0
    Prompt_context: str = ""

    @property
    def compression_ratio(self) -> float:
        if self.raw_repo_tokens == 0:
            return 1.0
        return 1.0 - (self.total_tokens_used / self.raw_repo_tokens)


class TopoLocalizer:
    """
    End-to-end Topological Code-Graph Localizer and Budgeted Subgraph Context Packer.
    Solves a constrained submodular coverage maximization problem under token budget B:
    - Top-K focal nodes receive full implementation source code
    - 1-hop structural neighbors receive AST-folded signatures + docstrings
    """

    def __init__(
        self,
        alpha: float = 0.25,
        token_budget: int = 2400,
        max_focal_entities: int = 5,
    ) -> None:
        self.alpha = alpha
        self.token_budget = token_budget
        self.max_focal_entities = max_focal_entities

    def localize(self, graph: CodeGraph, issue_text: str) -> BudgetedContextSlice:
        seed = extract_seed_distribution(graph, issue_text)
        ppr = personalized_pagerank(graph, seed, alpha=self.alpha)

        # Rank non-module entities first for focal selection
        candidates = [
            eid for eid in graph.entity_ids() if graph.entities[eid].kind != "module"
        ]
        ranked = sorted(candidates, key=lambda eid: ppr.get(eid, 0.0), reverse=True)

        raw_tokens = sum(
            e.full_token_estimate
            for e in graph.entities.values()
            if e.kind == "module"
        )

        focal: List[CodeEntity] = []
        focal_ids: set[str] = set()
        used_tokens = 0
        focal_budget = int(self.token_budget * 0.72)

        # Greedy density-ordered selection: maximize (ppr_score / token_cost^0.35)
        scored_candidates = sorted(
            ranked[:15],
            key=lambda eid: ppr.get(eid, 0.0)
            / (graph.entities[eid].full_token_estimate ** 0.25),
            reverse=True,
        )

        for eid in scored_candidates:
            if len(focal) >= self.max_focal_entities:
                break
            ent = graph.entities[eid]
            cost = ent.full_token_estimate
            if used_tokens + cost <= focal_budget or not focal:
                focal.append(ent)
                focal_ids.add(eid)
                used_tokens += cost

        # Collect 1-hop structural neighbors for AST skeleton folding
        neighbor_candidates: Dict[str, float] = {}
        for fid in focal_ids:
            for edge in graph.neighbors(fid, reverse=False) + graph.neighbors(
                fid, reverse=True
            ):
                other = edge.target if edge.source == fid else edge.source
                if other not in focal_ids and graph.entities[other].kind != "module":
                    neighbor_candidates[other] = max(
                        neighbor_candidates.get(other, 0.0),
                        ppr.get(other, 0.0) * edge.weight,
                    )

        sorted_neighbors = sorted(
            neighbor_candidates.keys(),
            key=lambda eid: neighbor_candidates[eid],
            reverse=True,
        )

        skeletons: List[CodeEntity] = []
        for nid in sorted_neighbors:
            ent = graph.entities[nid]
            cost = ent.skeleton_token_estimate
            if used_tokens + cost <= self.token_budget:
                skeletons.append(ent)
                used_tokens += cost

        prompt_blocks: List[str] = ["# === TOPOSWE FOCAL IMPLEMENTATIONS ==="]
        for ent in focal:
            prompt_blocks.append(
                f"# File: {ent.file_path} (Lines {ent.start_line}-{ent.end_line}) "
                f"[PPR={ppr.get(ent.entity_id, 0.0):.4f}]\n{ent.source_code}"
            )
        if skeletons:
            prompt_blocks.append("\n# === 1-HOP STRUCTURAL NEIGHBOR SIGNATURES (AST-FOLDED) ===")
            for ent in skeletons:
                prompt_blocks.append(
                    f"# File: {ent.file_path}::{ent.qualified_name}\n{ent.skeleton_code}"
                )

        return BudgetedContextSlice(
            focal_entities=focal,
            neighbor_skeletons=skeletons,
            ranked_entity_ids=ranked,
            ppr_scores=ppr,
            total_tokens_used=used_tokens,
            raw_repo_tokens=max(raw_tokens, used_tokens),
            Prompt_context="\n\n".join(prompt_blocks),
        )

