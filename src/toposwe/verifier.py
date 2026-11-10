"""
Deterministic Pre-Execution Syntax & Semantic Verification Gate (SSVG)
and Graph-Guided Impacted Test Selection.

Prevents multi-turn syntax regressions, indentation errors, undefined symbol
introductions, and accidental signature breakage before invoking costly test runs.
"""

from __future__ import annotations

import ast
import builtins
import textwrap
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from toposwe.graph import CodeGraph, EdgeType


_BUILTIN_NAMES: Set[str] = set(dir(builtins)) | {
    "self",
    "cls",
    "__name__",
    "__file__",
    "__doc__",
    "__all__",
    "Optional",
    "List",
    "Dict",
    "Set",
    "Tuple",
    "Any",
    "Union",
    "Callable",
    "Iterable",
}


@dataclass
class SearchReplaceBlock:
    """Structured SEARCH/REPLACE edit block targeting a single file."""

    file_path: str
    search_block: str
    replace_block: str


@dataclass
class PatchVerificationResult:
    """Deterministic diagnostic output produced by SyntaxSemanticGate."""

    passed: bool
    syntax_valid: bool
    undefined_symbols: List[str] = field(default_factory=list)
    signature_regressions: List[str] = field(default_factory=list)
    diagnostics: List[str] = field(default_factory=list)
    impacted_tests: List[str] = field(default_factory=list)
    patched_source: str = ""

    def format_feedback_prompt(self) -> str:
        """Compact diagnostic feedback injected back into the PatchSynthesizer subagent."""
        if self.passed:
            return "VERIFICATION_GATE: PASSED (0 syntax/semantic errors)."
        lines = ["VERIFICATION_GATE: REJECTED. Fix the following deterministic errors:"]
        for diag in self.diagnostics:
            lines.append(f"  - {diag}")
        return "\n".join(lines)


def apply_search_replace(
    original_source: str,
    edit: SearchReplaceBlock,
    auto_reindent: bool = True,
) -> Tuple[bool, str, str]:
    """
    Apply a SEARCH/REPLACE block with fallback whitespace-normalized matching
    and automatic indentation alignment.
    """
    if edit.search_block in original_source:
        count = original_source.count(edit.search_block)
        if count > 1:
            return (
                False,
                original_source,
                f"Ambiguous SEARCH block matches {count} locations in {edit.file_path}; include more surrounding context lines.",
            )
        patched = original_source.replace(edit.search_block, edit.replace_block, 1)
        return True, patched, ""

    if not auto_reindent:
        return (
            False,
            original_source,
            f"Exact SEARCH block not found in {edit.file_path}.",
        )

    # Fallback: line-by-line stripped comparison with indentation preservation
    orig_lines = original_source.splitlines()
    search_lines = [l for l in edit.search_block.splitlines() if l.strip()]
    if not search_lines:
        return False, original_source, "Empty SEARCH block."

    stripped_search = [l.strip() for l in search_lines]
    n_orig = len(orig_lines)
    matches: List[Tuple[int, int, int]] = []

    for i in range(n_orig):
        if orig_lines[i].strip() != stripped_search[0]:
            continue
        k = i
        matched = True
        for s_line in stripped_search:
            while k < n_orig and not orig_lines[k].strip():
                k += 1
            if k >= n_orig or orig_lines[k].strip() != s_line:
                matched = False
                break
            k += 1
        if matched:
            base_indent = len(orig_lines[i]) - len(orig_lines[i].lstrip())
            matches.append((i, k, base_indent))

    if len(matches) != 1:
        return (
            False,
            original_source,
            f"SEARCH block not uniquely matched in {edit.file_path} (found {len(matches)} fuzzy matches).",
        )

    start_idx, end_idx, target_indent = matches[0]
    dedented_replace = textwrap.dedent(edit.replace_block)
    reindented_lines = [
        (" " * target_indent + line if line.strip() else "")
        for line in dedented_replace.splitlines()
    ]
    new_lines = orig_lines[:start_idx] + reindented_lines + orig_lines[end_idx:]
    trailing_nl = "\n" if original_source.endswith("\n") else ""
    return True, "\n".join(new_lines) + trailing_nl, ""


class _ScopeAnalyzer(ast.NodeVisitor):
    """Lightweight static analyzer detecting undefined symbol references in Python AST."""

    def __init__(self) -> None:
        self.defined: Set[str] = set(_BUILTIN_NAMES)
        self.used_loads: List[Tuple[str, int]] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            bound = alias.asname or alias.name.split(".")[0]
            self.defined.add(bound)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name != "*":
                self.defined.add(alias.asname or alias.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.defined.add(node.name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.defined.add(node.name)
        for arg in (
            node.args.posonlyargs
            + node.args.args
            + node.args.kwonlyargs
            + ([node.args.vararg] if node.args.vararg else [])
            + ([node.args.kwarg] if node.args.kwarg else [])
        ):
            self.defined.add(arg.arg)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Global(self, node: ast.Global) -> None:
        for name in node.names:
            self.defined.add(name)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        for name in node.names:
            self.defined.add(name)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self.defined.add(node.name)
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        self._collect_targets(node.target)
        self.generic_visit(node)

    def _collect_targets(self, target: ast.AST) -> None:
        if isinstance(target, ast.Name):
            self.defined.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._collect_targets(elt)
        elif isinstance(target, ast.Starred):
            self._collect_targets(target.value)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Param)):
            self.defined.add(node.id)
        elif isinstance(node.ctx, ast.Load):
            self.used_loads.append((node.id, getattr(node, "lineno", 0)))


def _extract_public_arity(tree: ast.AST) -> Dict[str, Tuple[int, int, bool]]:
    """Map function/method qualified names to (min_positional, max_positional, has_varargs)."""
    signatures: Dict[str, Tuple[int, int, bool]] = {}
    for node in getattr(tree, "body", []):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            total_pos = len(node.args.posonlyargs) + len(node.args.args)
            n_defaults = len(node.args.defaults)
            min_pos = total_pos - n_defaults
            has_var = node.args.vararg is not None or node.args.kwarg is not None
            signatures[node.name] = (min_pos, total_pos, has_var)
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    total_pos = len(item.args.posonlyargs) + len(item.args.args)
                    n_defaults = len(item.args.defaults)
                    min_pos = total_pos - n_defaults
                    has_var = item.args.vararg is not None or item.args.kwarg is not None
                    signatures[f"{node.name}.{item.name}"] = (min_pos, total_pos, has_var)
    return signatures


class SyntaxSemanticGate:
    """
    Deterministic Syntax & Semantic Verification Gate (SSVG).
    Checks candidate patches in <5 ms before any container test execution:
      1. Exact or indentation-aligned SEARCH/REPLACE application
      2. Python AST syntax verification with exact line/column diagnostics
      3. Undefined symbol introduction check (new unbound Name loads)
      4. Public function/method arity compatibility check against callers
      5. Graph-guided impacted unit test selection
    """

    def __init__(self, check_signature_compat: bool = True) -> None:
        self.check_signature_compat = check_signature_compat

    def verify_patch(
        self,
        original_source: str,
        edit: SearchReplaceBlock,
        graph: Optional[CodeGraph] = None,
    ) -> PatchVerificationResult:
        applied, patched_source, apply_err = apply_search_replace(original_source, edit)
        if not applied:
            return PatchVerificationResult(
                passed=False,
                syntax_valid=False,
                diagnostics=[apply_err],
                patched_source=original_source,
            )

        diagnostics: List[str] = []
        try:
            new_tree = ast.parse(patched_source, filename=edit.file_path)
        except SyntaxError as exc:
            diag = (
                f"SyntaxError in {edit.file_path} at line {exc.lineno}, col {exc.offset}: "
                f"{exc.msg}"
            )
            return PatchVerificationResult(
                passed=False,
                syntax_valid=False,
                diagnostics=[diag],
                patched_source=patched_source,
            )

        try:
            old_tree = ast.parse(original_source, filename=edit.file_path)
        except SyntaxError:
            old_tree = ast.Module(body=[], type_ignores=[])

        # Check for newly introduced undefined symbols
        old_scope = _ScopeAnalyzer()
        old_scope.visit(old_tree)
        old_undef = {name for name, _ in old_scope.used_loads if name not in old_scope.defined}

        new_scope = _ScopeAnalyzer()
        new_scope.visit(new_tree)
        undefined_symbols: List[str] = []
        for name, lineno in new_scope.used_loads:
            if name not in new_scope.defined and name not in old_undef:
                if name not in undefined_symbols:
                    undefined_symbols.append(name)
                    diagnostics.append(
                        f"Undefined symbol '{name}' referenced at {edit.file_path}:{lineno}."
                    )

        # Check for accidental breaking changes in public callable signatures
        sig_regressions: List[str] = []
        if self.check_signature_compat:
            old_sigs = _extract_public_arity(old_tree)
            new_sigs = _extract_public_arity(new_tree)
            for fn_name, (old_min, old_max, _) in old_sigs.items():
                if fn_name.startswith("_") and fn_name != "__init__":
                    continue
                if fn_name not in new_sigs:
                    sig_regressions.append(fn_name)
                    diagnostics.append(
                        f"Public callable '{fn_name}' was removed from {edit.file_path}."
                    )
                else:
                    new_min, new_max, new_var = new_sigs[fn_name]
                    if new_min > old_min or (not new_var and new_max < old_max):
                        sig_regressions.append(fn_name)
                        diagnostics.append(
                            f"Signature regression on '{fn_name}': required positional args changed from [{old_min}..{old_max}] to [{new_min}..{new_max}] without default values."
                        )

        impacted_tests: List[str] = []
        if graph is not None:
            impacted_tests = self.select_impacted_tests(graph, edit.file_path)

        passed = len(diagnostics) == 0
        return PatchVerificationResult(
            passed=passed,
            syntax_valid=True,
            undefined_symbols=undefined_symbols,
            signature_regressions=sig_regressions,
            diagnostics=diagnostics,
            impacted_tests=impacted_tests,
            patched_source=patched_source,
        )

    @staticmethod
    def select_impacted_tests(graph: CodeGraph, modified_file: str) -> List[str]:
        """Traverse reverse CALLS/TESTS/IMPORTS edges to rank directly impacted test entities."""
        modified_nodes = [
            eid
            for eid, ent in graph.entities.items()
            if ent.file_path == modified_file and ent.kind != "module"
        ]
        visited: Set[str] = set(modified_nodes)
        frontier: List[str] = list(modified_nodes)
        test_files: List[str] = []

        for _ in range(3):  # Up to 3-hop reverse dependency traversal
            next_frontier: List[str] = []
            for nid in frontier:
                for edge in graph.neighbors(nid, reverse=True) + graph.neighbors(
                    nid, reverse=False
                ):
                    other = edge.source if edge.target == nid else edge.target
                    if other not in visited:
                        visited.add(other)
                        next_frontier.append(other)
                        ent = graph.entities[other]
                        if (
                            "test" in ent.file_path.lower()
                            or ent.name.startswith("test_")
                        ) and ent.file_path not in test_files:
                            test_files.append(ent.file_path)
            frontier = next_frontier

        return test_files

