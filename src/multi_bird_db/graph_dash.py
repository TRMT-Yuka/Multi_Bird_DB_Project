from __future__ import annotations

import argparse
import hashlib
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
    "other": "#8B8F97",
}

TAXON_RANK_LABELS = {
    "Q37517": "class",
    "Q5867051": "subclass",
    "Q2007442": "infraclass",
    "Q5868144": "superorder",
    "Q36602": "order",
    "Q5867959": "suborder",
    "Q2889003": "infraorder",
    "Q2136103": "superfamily",
    "Q35409": "family",
    "Q164280": "subfamily",
    "Q227936": "tribe",
    "Q3965313": "subtribe",
    "Q34740": "genus",
    "Q3238261": "subgenus",
    "Q6311258": "parvorder",
    "Q7432": "species",
    "Q68947": "subspecies",
    "Q279749": "form",
    "Q112082101": "ichnogenus",
}

TAXON_RANK_ORDER = [
    "Q37517",
    "Q5867051",
    "Q2007442",
    "Q5868144",
    "Q36602",
    "Q5867959",
    "Q2889003",
    "Q6311258",
    "Q2136103",
    "Q35409",
    "Q164280",
    "Q227936",
    "Q3965313",
    "Q34740",
    "Q3238261",
    "Q7432",
    "Q68947",
    "Q279749",
    "Q112082101",
]

LEGEND_ITEMS = [("other / missing", RANK_COLOR_MAP["other"])]

GRADIENT_ANCHORS = [
    "#E53935",  # red
    "#FDD835",  # yellow
    "#43A047",  # green
    "#1E88E5",  # blue
    "#FF4FA3",  # pink
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


def rank_key(node_data: dict[str, Any]) -> str:
    """Return the rank key used for color coding. / 色分けに使う rank キーを返す。"""

    rank_qid = str(node_data.get("taxon_rank") or "").strip()
    if rank_qid:
        return rank_qid
    rank_name = str(node_data.get("taxon_rank_name") or "").strip()
    return rank_name.lower() if rank_name else "other"


def rank_label(node_data: dict[str, Any]) -> str:
    """Return a human-readable rank label. / 人間向けの rank ラベルを返す。"""

    rank_qid = str(node_data.get("taxon_rank") or "").strip()
    if rank_qid:
        return TAXON_RANK_LABELS.get(rank_qid, rank_qid)
    rank_name = str(node_data.get("taxon_rank_name") or "").strip()
    return rank_name or "other / missing"


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[index : index + 2], 16) for index in (0, 2, 4))


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def _interpolate_color(start: str, end: str, t: float) -> str:
    start_rgb = _hex_to_rgb(start)
    end_rgb = _hex_to_rgb(end)
    blended = tuple(
        int(round(start_component + (end_component - start_component) * t))
        for start_component, end_component in zip(start_rgb, end_rgb, strict=True)
    )
    return _rgb_to_hex(blended)


def _gradient_color(index: int, total: int) -> str:
    if total <= 1:
        return GRADIENT_ANCHORS[0]
    anchor_count = len(GRADIENT_ANCHORS)
    position = index * (anchor_count - 1) / float(total - 1)
    left = int(position)
    right = min(left + 1, anchor_count - 1)
    local_t = position - left
    return _interpolate_color(GRADIENT_ANCHORS[left], GRADIENT_ANCHORS[right], local_t)


def _rank_order_index(rank_qid: str) -> int | None:
    try:
        return TAXON_RANK_ORDER.index(rank_qid)
    except ValueError:
        return None


def make_layout(root_qid: str) -> dict[str, Any]:
    """Build a Cytoscape layout config. / Cytoscape レイアウト設定を作る。"""

    return {
        "name": "breadthfirst",
        "directed": True,
        "roots": [root_qid],
        "fit": True,
        "padding": 90,
        "spacingFactor": 1.8,
        "circle": False,
    }


def rank_color(rank_key_value: str) -> str:
    """Map rank keys to an ordered gradient. / rank キーを順序付きグラデーションへ割り当てる。"""

    normalized = rank_key_value.strip()
    if not normalized:
        return RANK_COLOR_MAP["other"]
    order_index = _rank_order_index(normalized)
    if order_index is None:
        digest = hashlib.sha1(normalized.lower().encode("utf-8")).digest()
        seed = int.from_bytes(digest[:4], "big")
        # Unknown ranks stay gray-ish but still vary slightly so they are distinguishable.
        mix = 80 + (seed % 70)
        return _rgb_to_hex((mix, mix, mix))
    return _gradient_color(order_index, len(TAXON_RANK_ORDER))


def legend_block(items: list[tuple[str, str]]) -> html.Div:
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
                    for label, color in items
                ]
            ),
        ],
        style={
            "backgroundColor": "#fafafa",
            "padding": "12px",
            "border": "1px solid #d4d4d8",
        },
    )


def build_cytoscape_elements(graph: nx.DiGraph, positions: dict[str, dict[str, float]] | None = None) -> list[dict[str, Any]]:
    """Convert a NetworkX graph into Cytoscape elements. / NetworkX グラフを Cytoscape 要素へ変換する。"""

    positions = positions or {}
    elements: list[dict[str, Any]] = []
    for node_id, data in graph.nodes(data=True):
        rank_key_value = rank_key(data)
        rank_label_value = rank_label(data)
        position = positions.get(node_id)
        element: dict[str, Any] = {
            "data": {
                "id": node_id,
                "qid": node_id,
                "label": data.get("label_en") or node_id,
                "label_en": data.get("label_en") or node_id,
                "label_ja": data.get("label_ja") or "",
                "taxon_rank_key": rank_key_value,
                "taxon_rank": data.get("taxon_rank") or "",
                "taxon_rank_name": data.get("taxon_rank_name") or "",
                "taxon_rank_label": rank_label_value,
                "taxon_name": data.get("taxon_name") or "",
                "entity_url": data.get("entity_url") or "",
                "enwiki_url": data.get("enwiki_url") or "",
                "jawiki_url": data.get("jawiki_url") or "",
                "color": rank_color(rank_key_value),
            }
        }
        if position is not None:
            element["position"] = position
        elements.append(element)
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
        f"QID: {node_data.get('qid', node_data.get('id', ''))}",
        f"English label: {node_data.get('label_en', '')}",
        f"Taxon rank: {node_data.get('taxon_rank_label', node_data.get('taxon_rank_name', ''))} ({node_data.get('taxon_rank', '')})",
        f"Taxon name: {node_data.get('taxon_name', '')}",
    ]
    if node_data.get("entity_url"):
        lines.append(f"Wikidata: {node_data['entity_url']}")
    if node_data.get("enwiki_url"):
        lines.append(f"English Wikipedia: {node_data['enwiki_url']}")
    return "\n".join(lines)


def create_app(
    graph: nx.DiGraph,
    root_qid: str,
    max_depth: int,
    max_nodes: int,
) -> dash.Dash:
    """Create the Dash application. / Dash アプリを作る。"""

    app = dash.Dash(__name__)

    initial_subgraph = make_subgraph(graph, root_qid=root_qid, max_depth=max_depth, max_nodes=max_nodes)
    initial_elements = build_cytoscape_elements(initial_subgraph)
    legend_items = [
        (TAXON_RANK_LABELS[rank_qid], rank_color(rank_qid))
        for rank_qid in TAXON_RANK_ORDER
        if any(
            str(data.get("taxon_rank_key") or "").strip() == rank_qid
            for data in (element["data"] for element in initial_elements if "data" in element)
        )
    ]
    if not legend_items:
        legend_items = LEGEND_ITEMS
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
                    html.Div(id="legend-container", children=legend_block(legend_items)),
                ],
                style={"width": "320px", "padding": "16px", "borderRight": "1px solid #d4d4d8"},
            ),
            html.Div(
                [
                    cyto.Cytoscape(
                        id="taxonomy-graph",
                        elements=initial_elements,
                        style={"width": "100%", "height": "92vh"},
                        layout=make_layout(root_qid),
                        stylesheet=[
                            {
                                "selector": "node",
                                "style": {
                                    "label": "data(label)",
                                    "background-color": "data(color)",
                                    "text-wrap": "wrap",
                                    "text-max-width": "80px",
                                    "font-size": "8px",
                                    "color": "#111827",
                                    "border-width": 1,
                                    "border-color": "#111827",
                                    "width": 20,
                                    "height": 20,
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
        Output("taxonomy-graph", "layout"),
        Output("graph-summary", "children"),
        Output("legend-container", "children"),
        Input("update-graph", "n_clicks"),
        State("root-qid", "value"),
        State("max-depth", "value"),
        State("max-nodes", "value"),
    )
    def update_graph(
        _: int,
        selected_root_qid: str,
        selected_depth: int,
        selected_max_nodes: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], str, html.Div]:
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
        elements = build_cytoscape_elements(subgraph)
        subgraph_legend_items = [
            (TAXON_RANK_LABELS[rank_qid], rank_color(rank_qid))
            for rank_qid in TAXON_RANK_ORDER
            if any(
                str(data.get("taxon_rank_key") or "").strip() == rank_qid
                for data in (element["data"] for element in elements if "data" in element)
            )
        ]
        if not subgraph_legend_items:
            subgraph_legend_items = LEGEND_ITEMS
        return elements, make_layout(normalized_root), summary, legend_block(subgraph_legend_items)

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
    app = create_app(
        graph,
        root_qid=args.root_qid,
        max_depth=args.max_depth,
        max_nodes=args.max_nodes,
    )
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
