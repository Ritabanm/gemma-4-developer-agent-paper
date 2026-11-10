"""
Topology-Aware Direct Preference Optimization (Topo-DPO) and QLoRA Adapter
Calibration for Gemma 4 31B / 9B Developer Agents.

Formulates trajectory preference learning where chosen trajectories (tau^+)
satisfy both SSVG static verification and unit-test execution with minimal edit
locality, while rejected trajectories (tau^-) exhibit syntax/scope violations,
unlocalized edits, or redundant tool loops.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class PreferencePair:
    """A contrastive trajectory pair (x, y_w, y_l) with topological margin metadata."""

    issue_id: str
    prompt_tokens: int
    chosen_logprob_policy: float
    chosen_logprob_ref: float
    rejected_logprob_policy: float
    rejected_logprob_ref: float
    chosen_ppr_mass: float  # PPR probability mass of edited entities in y_w
    rejected_ppr_mass: float  # PPR probability mass of edited entities in y_l
    rejected_had_syntax_error: bool = False
    chosen_turn_count: int = 2
    rejected_turn_count: int = 5


@dataclass
class TopoDPOLossConfig:
    """Hyperparameters for Topology-Aware DPO and LoRA rank-16 post-training."""

    beta: float = 0.12
    lambda_topo: float = 0.35
    gamma_gate: float = 0.25
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def compute_topo_dpo_loss_single(
    pair: PreferencePair,
    config: TopoDPOLossConfig,
) -> Tuple[float, float, float]:
    """
    Compute the Topology-Aware DPO loss for a single contrastive pair:
        L_TopoDPO = - w(tau^+, tau^-) * log sigma(
            beta * [(log pi_theta(y_w|x) - log pi_ref(y_w|x))
                  - (log pi_theta(y_l|x) - log pi_ref(y_l|x))]
            - m_topo(y_w, y_l)
        )
    Returns (loss, implicit_reward_margin, preference_accuracy).
    """
    chosen_ratio = pair.chosen_logprob_policy - pair.chosen_logprob_ref
    rejected_ratio = pair.rejected_logprob_policy - pair.rejected_logprob_ref
    raw_margin = config.beta * (chosen_ratio - rejected_ratio)

    # Topological & verification margin shift
    ppr_delta = max(0.0, pair.chosen_ppr_mass - pair.rejected_ppr_mass)
    gate_bonus = config.gamma_gate if pair.rejected_had_syntax_error else 0.0
    topo_margin = config.lambda_topo * ppr_delta + gate_bonus

    # Importance weight emphasizing hard syntax/scope regressions and high-PPR edits
    weight = 1.0 + 0.5 * ppr_delta + (0.35 if pair.rejected_had_syntax_error else 0.0)
    logit = raw_margin + topo_margin
    prob = _sigmoid(logit)
    loss = -weight * math.log(max(prob, 1e-12))
    acc = 1.0 if raw_margin > 0.0 else 0.0
    return loss, raw_margin, acc


def compute_topo_dpo_metrics(
    pairs: List[PreferencePair],
    config: TopoDPOLossConfig,
) -> Dict[str, float]:
    """Evaluate aggregate Topo-DPO loss, reward margin, and preference accuracy."""
    if not pairs:
        return {"loss": 0.0, "reward_margin": 0.0, "preference_accuracy": 0.0}

    losses: List[float] = []
    margins: List[float] = []
    accs: List[float] = []

    for pair in pairs:
        l_val, m_val, a_val = compute_topo_dpo_loss_single(pair, config)
        losses.append(l_val)
        margins.append(m_val)
        accs.append(a_val)

    return {
        "loss": float(np.mean(losses)),
        "reward_margin": float(np.mean(margins)),
        "preference_accuracy": float(np.mean(accs)),
        "n_pairs": len(pairs),
    }


def filter_trajectories_for_dpo(
    candidates: List[PreferencePair],
    min_ppr_advantage: float = 0.05,
) -> List[PreferencePair]:
    """Filter raw rollout trajectories to retain high-signal contrastive DPO pairs."""
    filtered: List[PreferencePair] = []
    for p in candidates:
        ppr_adv = p.chosen_ppr_mass - p.rejected_ppr_mass
        if p.rejected_had_syntax_error or ppr_adv >= min_ppr_advantage:
            if p.chosen_turn_count <= p.rejected_turn_count:
                filtered.append(p)
    return filtered

