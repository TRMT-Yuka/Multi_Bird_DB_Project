from __future__ import annotations

import argparse
import pickle
from collections import deque
from pathlib import Path
from typing import Any

import dash
import dash_cytoscape as cyto
import networkx as nx
from dash import Input, Output, State, dcc, html

from .config import get_project_paths

RANK_COLOR_MAP = {
    "class": "#4D8DFF",
    "order": "#00FF00",
    "family": "#FFFF00",
    "genus": "#FFA500",
    "species": "#FF0000",
    "other": "#8B8F97",
}

LEGEND_ITEMS = [
    ("class", RANK_COLOR_MAP["class"]),
    ("order", RANK_COLOR_MAP["order"]),
    ("family", RANK_COLOR_MAP["family"]),
    ("genus", RANK_COLOR_MAP["genus"]),
    ("species", RANK_COLOR_MAP["species"]),
    ("other / missing", RANK_COLOR_MAP["other"]),
]


def load_graph(graph_path: Path) -> nx.DiGraph:
    """Load a pickled NetworkX graph. / pickle 化された NetworkX グラフを読む。"""

    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file does not exist: {graph_path}")
    with graph_path.open("rb") as handle:
        graph = pickle.load(handle)
    if not isinstance(graph, nx.DiGraph):
        raise ValueError(f"Graph PKL must contain a networkx.DiGraph, got: {type(graph).__name__}")
    return graph


def collect_descendants(graph: nx.DiGraph, root_qid: str, max_depth: int, max_nodes: int) -> list[str]:
    """Collect descendants in BFS order for a bounded visualization. / 可視化用に深さ制限付きで子孫ノードを集める。"""

    if root_qid not in graph:
        raise KeyError(f"QID not found in graph: {root_qid}")

    ordered: list[str] = []
    visited: set[str] = set()
    queue: deque[tuple[str, int]] = deque([(root_qid, 0)])

    while queue and len(ordered) < max_nodes:
        node, depth = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        ordered.append(node)
        if depth >= max_depth:
            continue
        for child in sorted(graph.successors(node)):
            if child not in visited:
                queue.append((child, depth + 1))
    return ordered


def make_subgraph(graph: nx.DiGraph, root_qid: str, max_depth: int, max_nodes: int) -> nx.DiGraph:
    """Create the visualization subgraph. / 可視化対象の部分グラフを作る。"""

    nodes = collect_descendants(graph, root_qid=root_qid, max_depth=max_depth, max_nodes=max_nodes)
    return graph.subgraph(nodes).copy()


def rank_color(rank_name: str) -> str:
    """Map taxon rank names to stable colors. / taxon rank 名から色を割り当てる。"""

    normalized = rank_name.strip().lower()
    return RANK_COLOR_MAP.get(normalized, RANK_COLOR_MAP["other"])


def legend_block() -> html.Div:
    """Create the color legend block. / 色凡例ブロックを作る。"""

    return html.Div(
        [
            html.H3("Legend", style={"marginBottom": "10px", "marginTop": "0"}),
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                style={
                                    "display": "inline-block",
                                    "width": "14px",
                                    "height": "14px",
                                    "backgroundColor": color,
                                    "border": "1px solid #111827",
                                    "marginRight": "8px",
                                    "verticalAlign": "middle",
                                }
                            ),
                            html.Span(label, style={"verticalAlign": "middle"}),
                        ],
                        style={"marginBottom": "6px"},
                    )
                    for label, color in LEGEND_ITEMS
                ]
            ),
        ],
        style={
            "backgroundColor": "#fafafa",
            "padding": "12px",
            "border": "1px solid #d4d4d8",
        },
    )


def build_cytoscape_elements(graph: nx.DiGraph) -> list[dict[str, Any]]:
    """Convert a NetworkX graph into Cytoscape elements. / NetworkX グラフを Cytoscape 要素へ変換する。"""

    elements: list[dict[str, Any]] = []
    for node_id, data in graph.nodes(data=True):
        rank_name = str(data.get("taxon_rank_name") or "")
        elements.append(
            {
                "data": {
                    "id": node_id,
                    "label": data.get("label_en") or node_id,
                    "label_en": data.get("label_en") or node_id,
                    "label_ja": data.get("label_ja") or "",
                    "taxon_rank": data.get("taxon_rank") or "",
                    "taxon_rank_name": rank_name,
                    "taxon_name": data.get("taxon_name") or "",
                    "entity_url": data.get("entity_url") or "",
                    "enwiki_url": data.get("enwiki_url") or "",
                    "jawiki_url": data.get("jawiki_url") or "",
                    "color": rank_color(rank_name),
                }
            }
        )
    for source, target, data in graph.edges(data=True):
        elements.append(
            {
                "data": {
                    "id": f"{source}->{target}",
                    "source": source,
                    "target": target,
                    "relation": data.get("relation") or "parent_taxon",
                }
            }
        )
    return elements


def format_node_details(node_data: dict[str, Any] | None) -> str:
    """Format the clicked node metadata for the side panel. / クリックしたノードの詳細を整形する。"""

    if not node_data:
        return "Click a node to inspect its metadata."
    lines = [
        f"QID: {node_data.get('id', '')}",
        f"English label: {node_data.get('label_en', '')}",
        f"Taxon rank: {node_data.get('taxon_rank_name', '')} ({node_data.get('taxon_rank', '')})",
        f"Taxon name: {node_data.get('taxon_name', '')}",
    ]
    if node_data.get("entity_url"):
        lines.append(f"Wikidata: {node_data['entity_url']}")
    if node_data.get("enwiki_url"):
        lines.append(f"English Wikipedia: {node_data['enwiki_url']}")
    return "\n".join(lines)


def create_app(graph: nx.DiGraph, root_qid: str, max_depth: int, max_nodes: int) -> dash.Dash:
    """Create the Dash application. / Dash アプリを作る。"""

    app = dash.Dash(__name__)

    initial_subgraph = make_subgraph(graph, root_qid=root_qid, max_depth=max_depth, max_nodes=max_nodes)
    initial_elements = build_cytoscape_elements(initial_subgraph)
    initial_info = (
        f"Root QID: {root_qid}\n"
        f"Nodes shown: {initial_subgraph.number_of_nodes()}\n"
        f"Edges shown: {initial_subgraph.number_of_edges()}\n"
        "Click a node to inspect its metadata."
    )

    app.layout = html.Div(
        [
            html.Div(
                [
                    html.H1("Bird Taxonomy Graph Explorer"),
                    html.Div(
                        [
                            html.Label("Root QID"),
                            dcc.Input(id="root-qid", type="text", value=root_qid, debounce=True),
                            html.Label("Max depth"),
                            dcc.Input(id="max-depth", type="number", value=max_depth, min=0, step=1),
                            html.Label("Max nodes"),
                            dcc.Input(id="max-nodes", type="number", value=max_nodes, min=1, step=1),
                            html.Button("Update", id="update-graph", n_clicks=0),
                        ],
                        style={
                            "display": "grid",
                            "gridTemplateColumns": "140px 180px",
                            "gap": "10px 12px",
                            "alignItems": "center",
                            "maxWidth": "460px",
                        },
                    ),
                    html.Pre(
                        initial_info,
                        id="graph-summary",
                        style={
                            "whiteSpace": "pre-wrap",
                            "backgroundColor": "#f4f4f5",
                            "padding": "12px",
                            "border": "1px solid #d4d4d8",
                        },
                    ),
                    html.Pre(
                        "Click a node to inspect its metadata.",
                        id="node-details",
                        style={
                            "whiteSpace": "pre-wrap",
                            "backgroundColor": "#fafafa",
                            "padding": "12px",
                            "border": "1px solid #d4d4d8",
                            "minHeight": "200px",
                        },
                    ),
                    legend_block(),
                ],
                style={"width": "320px", "padding": "16px", "borderRight": "1px solid #d4d4d8"},
            ),
            html.Div(
                [
                    cyto.Cytoscape(
                        id="taxonomy-graph",
                        elements=initial_elements,
                        style={"width": "100%", "height": "92vh"},
                        layout={"name": "breadthfirst", "directed": True, "padding": 18, "spacingFactor": 1.0},
                        stylesheet=[
                            {
                                "selector": "node",
                                "style": {
                                    "label": "data(label)",
                                    "background-color": "data(color)",
                                    "text-wrap": "wrap",
                                    "text-max-width": "90px",
                                    "font-size": "9px",
                                    "color": "#111827",
                                    "border-width": 1,
                                    "border-color": "#111827",
                                    "width": 28,
                                    "height": 28,
                                },
                            },
                            {
                                "selector": "edge",
                                "style": {
                                    "curve-style": "bezier",
                                    "target-arrow-shape": "triangle",
                                    "target-arrow-color": "#6b7280",
                                    "line-color": "#9ca3af",
                                    "width": 1.3,
                                },
                            },
                            {
                                "selector": ":selected",
                                "style": {
                                    "border-color": "#b91c1c",
                                    "border-width": 3,
                                },
                            },
                        ],
                    )
                ],
                style={"flex": "1", "minWidth": 0},
            ),
        ],
        style={"display": "flex", "fontFamily": "Arial, sans-serif"},
    )

    @app.callback(
        Output("taxonomy-graph", "elements"),
        Output("graph-summary", "children"),
        Input("update-graph", "n_clicks"),
        State("root-qid", "value"),
        State("max-depth", "value"),
        State("max-nodes", "value"),
    )
    def update_graph(_: int, selected_root_qid: str, selected_depth: int, selected_max_nodes: int) -> tuple[list[dict[str, Any]], str]:
        normalized_root = (selected_root_qid or root_qid).strip()
        normalized_depth = int(selected_depth or max_depth)
        normalized_max_nodes = int(selected_max_nodes or max_nodes)
        subgraph = make_subgraph(
            graph,
            root_qid=normalized_root,
            max_depth=normalized_depth,
            max_nodes=normalized_max_nodes,
        )
        summary = (
            f"Root QID: {normalized_root}\n"
            f"Nodes shown: {subgraph.number_of_nodes()}\n"
            f"Edges shown: {subgraph.number_of_edges()}\n"
            f"Max depth: {normalized_depth}\n"
            f"Max nodes: {normalized_max_nodes}"
        )
        return build_cytoscape_elements(subgraph), summary

    @app.callback(Output("node-details", "children"), Input("taxonomy-graph", "tapNodeData"))
    def show_node_details(node_data: dict[str, Any] | None) -> str:
        return format_node_details(node_data)

    return app


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the Dash graph viewer. / Dash graph viewer 用の引数パーサを作る。"""

    paths = get_project_paths()
    parser = argparse.ArgumentParser(description="Serve an interactive Dash Cytoscape viewer for the taxonomy graph.")
    parser.add_argument("--input", default=str(paths.taxonomy_graph_pkl))
    parser.add_argument("--root-qid", default="Q5113")
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-nodes", type=int, default=150)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the Dash graph viewer. / Dash graph viewer を起動する。"""

    args = build_parser().parse_args(argv)
    graph = load_graph(Path(args.input))
    app = create_app(graph, root_qid=args.root_qid, max_depth=args.max_depth, max_nodes=args.max_nodes)
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
