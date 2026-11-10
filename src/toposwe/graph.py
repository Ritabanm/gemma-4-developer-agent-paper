"""
Heterogeneous AST-based Repository Code Graph Construction and Skeletonization.

Parses Python repositories into a directed multigraph G = (V, E, tau_V, tau_E)
where vertices represent modules, classes, functions, and methods, and typed
edges encode structural containment, symbol imports, class inheritance,
function/method invocations, and exception propagation.
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


class EdgeType(str, Enum):
    """Typed structural edges in the repository code graph."""

    CONTAINS = "contains"
    IMPORTS = "imports"
    INHERITS = "inherits"
    CALLS = "calls"
    RAISES = "raises"
    TESTS = "tests"


EDGE_WEIGHTS: Dict[EdgeType, float] = {
    EdgeType.CALLS: 1.0,
    EdgeType.INHERITS: 0.85,
    EdgeType.IMPORTS: 0.65,
    EdgeType.CONTAINS: 0.45,
    EdgeType.RAISES: 0.75,
    EdgeType.TESTS: 0.80,
}


@dataclass
class CodeEntity:
    """A node v in V representing an addressable code unit in the repository."""

    entity_id: str  # e.g. "pkg/module.py::ClassName.method_name"
    name: str
    qualified_name: str
    file_path: str
    kind: str  # "module" | "class" | "function" | "method"
    start_line: int
    end_line: int
    signature: str
    docstring: str
    source_code: str
    skeleton_code: str
    called_symbols: List[str] = field(default_factory=list)
    imported_symbols: List[str] = field(default_factory=list)
    base_classes: List[str] = field(default_factory=list)
    raised_exceptions: List[str] = field(default_factory=list)

    @property
    def full_token_estimate(self) -> int:
        """Approximate token count of full implementation (~4 chars/token)."""
        return max(1, len(self.source_code) // 4 + 4)

    @property
    def skeleton_token_estimate(self) -> int:
        """Approximate token count of AST-folded signature + docstring."""
        return max(1, len(self.skeleton_code) // 4 + 2)


@dataclass
class GraphEdge:
    """Directed weighted edge (u -> v) with relation type tau_E."""

    source: str
    target: str
    edge_type: EdgeType
    weight: float


class CodeGraph:
    """Directed heterogeneous repository dependency graph."""

    def __init__(self) -> None:
        self.entities: Dict[str, CodeEntity] = {}
        self.edges: List[GraphEdge] = []
        self._adj: Dict[str, List[GraphEdge]] = {}
        self._rev_adj: Dict[str, List[GraphEdge]] = {}
        self._symbol_index: Dict[str, List[str]] = {}

    def add_entity(self, entity: CodeEntity) -> None:
        self.entities[entity.entity_id] = entity
        self._adj.setdefault(entity.entity_id, [])
        self._rev_adj.setdefault(entity.entity_id, [])
        for key in {entity.name, entity.qualified_name, entity.entity_id}:
            self._symbol_index.setdefault(key, [])
            if entity.entity_id not in self._symbol_index[key]:
                self._symbol_index[key].append(entity.entity_id)

    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: EdgeType,
        weight: Optional[float] = None,
    ) -> None:
        if source not in self.entities or target not in self.entities:
            return
        if source == target:
            return
        w = weight if weight is not None else EDGE_WEIGHTS.get(edge_type, 0.5)
        edge = GraphEdge(source=source, target=target, edge_type=edge_type, weight=w)
        self.edges.append(edge)
        self._adj[source].append(edge)
        self._rev_adj[target].append(edge)

    def resolve_symbol(self, symbol: str) -> List[str]:
        """Resolve an unqualified or partially qualified name to entity IDs."""
        if symbol in self._symbol_index:
            return list(self._symbol_index[symbol])
        tail = symbol.split(".")[-1]
        return list(self._symbol_index.get(tail, []))

    def neighbors(self, entity_id: str, reverse: bool = False) -> List[GraphEdge]:
        if reverse:
            return list(self._rev_adj.get(entity_id, []))
        return list(self._adj.get(entity_id, []))

    def entity_ids(self) -> List[str]:
        return list(self.entities.keys())


def _extract_signature(node: ast.AST, lines: List[str]) -> str:
    """Extract clean header signature of a class or function/method."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        prefix = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
        args_str = ast.unparse(node.args) if hasattr(ast, "unparse") else "..."
        ret_str = ""
        if node.returns is not None and hasattr(ast, "unparse"):
            ret_str = f" -> {ast.unparse(node.returns)}"
        decorators = [
            f"@{ast.unparse(d)}" for d in node.decorator_list if hasattr(ast, "unparse")
        ]
        dec_prefix = ("\n".join(decorators) + "\n") if decorators else ""
        return f"{dec_prefix}{prefix}{node.name}({args_str}){ret_str}:"
    if isinstance(node, ast.ClassDef):
        bases = [ast.unparse(b) for b in node.bases] if hasattr(ast, "unparse") else []
        base_str = f"({', '.join(bases)})" if bases else ""
        return f"class {node.name}{base_str}:"
    return lines[getattr(node, "lineno", 1) - 1].strip()


def _make_skeleton(signature: str, docstring: str, indent: str = "") -> str:
    """Create a compact AST-folded signature + single-line docstring stub."""
    indented_sig = "\n".join(f"{indent}{line}" for line in signature.splitlines())
    if docstring:
        first_line = docstring.strip().splitlines()[0].strip()
        return f'{indented_sig}\n{indent}    """{first_line}"""\n{indent}    ...'
    return f"{indented_sig}\n{indent}    ..."


class _EntityVisitor(ast.NodeVisitor):
    """Collects calls, imports, and raised exceptions inside an AST subtree."""

    def __init__(self) -> None:
        self.calls: List[str] = []
        self.imports: List[str] = []
        self.raises: List[str] = []

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            self.calls.append(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                self.calls.append(f"{node.func.value.id}.{node.func.attr}")
            self.calls.append(node.func.attr)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append(alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        for alias in node.names:
            self.imports.append(f"{mod}.{alias.name}" if mod else alias.name)
            self.imports.append(alias.name)

    def visit_Raise(self, node: ast.Raise) -> None:
        if isinstance(node.exc, ast.Call) and isinstance(node.exc.func, ast.Name):
            self.raises.append(node.exc.func.id)
        elif isinstance(node.exc, ast.Name):
            self.raises.append(node.exc.id)
        self.generic_visit(node)


def parse_python_file(file_path: str, rel_path: str) -> List[CodeEntity]:
    """Parse a single Python file into module, class, function, and method entities."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        source = f.read()
    lines = source.splitlines()
    try:
        tree = ast.parse(source, filename=rel_path)
    except SyntaxError:
        return []

    entities: List[CodeEntity] = []
    mod_doc = ast.get_docstring(tree) or ""
    mod_visitor = _EntityVisitor()
    mod_visitor.visit(tree)

    top_Wait_lines: List[str] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)) and hasattr(ast, "unparse"):
            top_Wait_lines.append(ast.unparse(node))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            top_Wait_lines.append(_extract_signature(node, lines) + " ...")

    mod_skeleton = "\n".join(top_Wait_lines) if top_Wait_lines else f"# Module {rel_path}"
    mod_id = f"{rel_path}::<module>"
    entities.append(
        CodeEntity(
            entity_id=mod_id,
            name=Path(rel_path).stem,
            qualified_name=rel_path.replace("/", ".").removesuffix(".py"),
            file_path=rel_path,
            kind="module",
            start_line=1,
            end_line=max(1, len(lines)),
            signature=f"# Module: {rel_path}",
            docstring=mod_doc,
            source_code=source,
            skeleton_code=mod_skeleton,
            called_symbols=list(dict.fromkeys(mod_visitor.calls)),
            imported_symbols=list(dict.fromkeys(mod_visitor.imports)),
            raised_exceptions=list(dict.fromkeys(mod_visitor.raises)),
        )
    )

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start, end = node.lineno, getattr(node, "end_lineno", node.lineno)
            fn_src = "\n".join(lines[start - 1 : end])
            sig = _extract_signature(node, lines)
            doc = ast.get_docstring(node) or ""
            vis = _EntityVisitor()
            vis.visit(node)
            entities.append(
                CodeEntity(
                    entity_id=f"{rel_path}::{node.name}",
                    name=node.name,
                    qualified_name=node.name,
                    file_path=rel_path,
                    kind="function",
                    start_line=start,
                    end_line=end,
                    signature=sig,
                    docstring=doc,
                    source_code=fn_src,
                    skeleton_code=_make_skeleton(sig, doc),
                    called_symbols=list(dict.fromkeys(vis.calls)),
                    imported_symbols=list(dict.fromkeys(vis.imports)),
                    raised_exceptions=list(dict.fromkeys(vis.raises)),
                )
            )
        elif isinstance(node, ast.ClassDef):
            c_start, c_end = node.lineno, getattr(node, "end_lineno", node.lineno)
            c_src = "\n".join(lines[c_start - 1 : c_end])
            c_sig = _extract_signature(node, lines)
            c_doc = ast.get_docstring(node) or ""
            c_vis = _EntityVisitor()
            c_vis.visit(node)
            bases = [ast.unparse(b) for b in node.bases] if hasattr(ast, "unparse") else []
            method_stubs: List[str] = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    m_start = item.lineno
                    m_end = getattr(item, "end_lineno", item.lineno)
                    m_src = "\n".join(lines[m_start - 1 : m_end])
                    m_sig = _extract_signature(item, lines)
                    m_doc = ast.get_docstring(item) or ""
                    m_vis = _EntityVisitor()
                    m_vis.visit(item)
                    method_stubs.append(_make_skeleton(m_sig, m_doc, indent="    "))
                    entities.append(
                        CodeEntity(
                            entity_id=f"{rel_path}::{node.name}.{item.name}",
                            name=item.name,
                            qualified_name=f"{node.name}.{item.name}",
                            file_path=rel_path,
                            kind="method",
                            start_line=m_start,
                            end_line=m_end,
                            signature=m_sig,
                            docstring=m_doc,
                            source_code=m_src,
                            skeleton_code=_make_skeleton(m_sig, m_doc, indent="    "),
                            called_symbols=list(dict.fromkeys(m_vis.calls)),
                            imported_symbols=list(dict.fromkeys(m_vis.imports)),
                            raised_exceptions=list(dict.fromkeys(m_vis.raises)),
                        )
                    )
            cls_skel = c_sig + "\n" + ("\n".join(method_stubs) if method_stubs else "    ...")
            entities.append(
                CodeEntity(
                    entity_id=f"{rel_path}::{node.name}",
                    name=node.name,
                    qualified_name=node.name,
                    file_path=rel_path,
                    kind="class",
                    start_line=c_start,
                    end_line=c_end,
                    signature=c_sig,
                    docstring=c_doc,
                    source_code=c_src,
                    skeleton_code=cls_skel,
                    called_symbols=list(dict.fromkeys(c_vis.calls)),
                    imported_symbols=list(dict.fromkeys(c_vis.imports)),
                    base_classes=bases,
                    raised_exceptions=list(dict.fromkeys(c_vis.raises)),
                )
            )

    return entities


def build_repository_graph(
    repo_root: str,
    ignore_dirs: Optional[Set[str]] = None,
) -> CodeGraph:
    """Construct the complete heterogeneous CodeGraph across a repository."""
    if ignore_dirs is None:
        ignore_dirs = {".git", "__pycache__", ".venv", "venv", "node_modules", ".pytest_cache"}

    graph = CodeGraph()
    root_path = Path(repo_root).resolve()

    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames[:] = [d for d in sorted(dirnames) if d not in ignore_dirs]
        for fname in sorted(filenames):
            if not fname.endswith(".py"):
                continue
            abs_p = os.path.join(dirpath, fname)
            rel_p = os.path.relpath(abs_p, root_path).replace(os.sep, "/")
            for entity in parse_python_file(abs_p, rel_p):
                graph.add_entity(entity)

    # Wire structural, call, import, inheritance, and test edges
    for eid, ent in graph.entities.items():
        mod_id = f"{ent.file_path}::<module>"
        if eid != mod_id and mod_id in graph.entities:
            graph.add_edge(mod_id, eid, EdgeType.CONTAINS)
            graph.add_edge(eid, mod_id, EdgeType.CONTAINS, weight=0.25)

        if ent.kind == "method":
            cls_name = ent.qualified_name.split(".")[0]
            cls_id = f"{ent.file_path}::{cls_name}"
            if cls_id in graph.entities:
                graph.add_edge(cls_id, eid, EdgeType.CONTAINS)
                graph.add_edge(eid, cls_id, EdgeType.CONTAINS, weight=0.35)

        for sym in ent.called_symbols:
            for target_id in graph.resolve_symbol(sym):
                if graph.entities[target_id].kind in {"function", "method", "class"}:
                    is_test = "test" in ent.file_path.lower() or ent.name.startswith("test_")
                    etype = EdgeType.TESTS if is_test else EdgeType.CALLS
                    graph.add_edge(eid, target_id, etype)
                    # Bidirectional coupling (caller <-> callee fault propagation)
                    graph.add_edge(target_id, eid, etype, weight=0.6 * EDGE_WEIGHTS[etype])

        for imp in ent.imported_symbols:
            for target_id in graph.resolve_symbol(imp):
                graph.add_edge(eid, target_id, EdgeType.IMPORTS)
                graph.add_edge(target_id, eid, EdgeType.IMPORTS, weight=0.35)

        for base in ent.base_classes:
            for target_id in graph.resolve_symbol(base):
                if graph.entities[target_id].kind == "class":
                    graph.add_edge(eid, target_id, EdgeType.INHERITS)
                    graph.add_edge(target_id, eid, EdgeType.INHERITS, weight=0.5)

        for exc in ent.raised_exceptions:
            for target_id in graph.resolve_symbol(exc):
                graph.add_edge(eid, target_id, EdgeType.RAISES)
                graph.add_edge(target_id, eid, EdgeType.RAISES, weight=0.5)

    return graph

