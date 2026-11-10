"""Unit test suite for TopoSWE graph construction, PPR localization, SSVG gate, and Topo-DPO."""

from __future__ import annotations

import os
import tempfile

from toposwe.agent import SubagentHandoffState, TopoSWEOrchestrator
from toposwe.graph import EdgeType, build_repository_graph
from toposwe.localizer import TopoLocalizer, compute_bm25_scores, personalized_pagerank
from toposwe.trainer import (
    PreferencePair,
    TopoDPOLossConfig,
    compute_topo_dpo_metrics,
    filter_trajectories_for_dpo,
)
from toposwe.verifier import SearchReplaceBlock, SyntaxSemanticGate


def _make_sample_repo(root: str) -> str:
    pkg = os.path.join(root, "billing")
    tests = os.path.join(root, "tests")
    os.makedirs(pkg, exist_ok=True)
    os.makedirs(tests, exist_ok=True)

    tax_src = (
        "class TaxCalculator:\n"
        '    """Computes regional tax rates."""\n'
        "    def compute_tax(self, subtotal: float, rate: float = None) -> float:\n"
        "        return subtotal * rate\n"
    )
    with open(os.path.join(pkg, "tax.py"), "w", encoding="utf-8") as f:
        f.write(tax_src)

    inv_src = (
        "from billing.tax import TaxCalculator\n\n"
        "class InvoiceProcessor:\n"
        "    def finalize_invoice(self, amount: float) -> float:\n"
        "        calc = TaxCalculator()\n"
        "        return amount + calc.compute_tax(amount, None)\n"
    )
    with open(os.path.join(pkg, "invoice.py"), "w", encoding="utf-8") as f:
        f.write(inv_src)

    test_src = (
        "from billing.invoice import InvoiceProcessor\n\n"
        "def test_finalize_invoice_default_rate():\n"
        "    proc = InvoiceProcessor()\n"
        "    assert proc.finalize_invoice(100.0) == 100.0\n"
    )
    with open(os.path.join(tests, "test_invoice.py"), "w", encoding="utf-8") as f:
        f.write(test_src)

    return tax_src


def test_graph_and_ppr_localization() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _make_sample_repo(tmp)
        graph = build_repository_graph(tmp)
        assert "billing/tax.py::TaxCalculator.compute_tax" in graph.entities
        assert "billing/invoice.py::InvoiceProcessor.finalize_invoice" in graph.entities

        issue = (
            "TypeError in finalize_invoice when rate is None.\n"
            '  File "billing/invoice.py", line 6, in finalize_invoice\n'
            '  File "billing/tax.py", line 4, in compute_tax'
        )
        localizer = TopoLocalizer(alpha=0.25, token_budget=500, max_focal_entities=2)
        ctx = localizer.localize(graph, issue)
        focal_ids = [e.entity_id for e in ctx.focal_entities]
        assert "billing/tax.py::TaxCalculator.compute_tax" in focal_ids
        assert ctx.total_tokens_used <= 500


def test_syntax_semantic_gate_catches_errors() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tax_src = _make_sample_repo(tmp)
        graph = build_repository_graph(tmp)
        gate = SyntaxSemanticGate()

        # 1. Undefined symbol regression should be rejected
        bad_undef = SearchReplaceBlock(
            file_path="billing/tax.py",
            search_block="return subtotal * rate",
            replace_block="return subtotal * undefined_default_rate(rate)",
        )
        res_undef = gate.verify_patch(tax_src, bad_undef, graph=graph)
        assert not res_undef.passed
        assert "undefined_default_rate" in res_undef.undefined_symbols

        # 2. Valid patch should pass and identify impacted test file
        good_edit = SearchReplaceBlock(
            file_path="billing/tax.py",
            search_block="return subtotal * rate",
            replace_block="return subtotal * (rate or 0.0)",
        )
        res_good = gate.verify_patch(tax_src, good_edit, graph=graph)
        assert res_good.passed
        assert "tests/test_invoice.py" in res_good.impacted_tests


def test_topo_dpo_metrics() -> None:
    pair = PreferencePair(
        issue_id="test_01",
        prompt_tokens=1200,
        chosen_logprob_policy=-10.0,
        chosen_logprob_ref=-14.0,
        rejected_logprob_policy=-16.0,
        rejected_logprob_ref=-13.0,
        chosen_ppr_mass=0.55,
        rejected_ppr_mass=0.10,
        rejected_had_syntax_error=True,
    )
    filtered = filter_trajectories_for_dpo([pair])
    assert len(filtered) == 1
    metrics = compute_topo_dpo_metrics(filtered, TopoDPOLossConfig())
    assert metrics["preference_accuracy"] == 1.0
    assert metrics["loss"] > 0.0

