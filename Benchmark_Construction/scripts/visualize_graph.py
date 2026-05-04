"""Render a pubmed_graph GraphML in the dense "hairball" style with a
left-side entity-type legend and a red-circled zoomed-in inset around
the top-degree hub.

Usage:
  python scripts/visualize_graph.py \
      --graphml benchmark_runs/proteinlmbench_full_sciverse/global_graph.graphml \
      --output  figures/base_graph.png \
      --title   "Protein Knowledge Graph (base)"

Design notes:
  - super-hubs (deg > --hub-cap) get their edges trimmed to the top-K
    neighbours for the LAYOUT only, so spring_layout doesn't collapse.
    Full graph stats are reported on stdout before trimming.
  - node colour = PALETTE[node_type]; palette is stable across graphs
    so the same entity type renders the same colour in BASE and V3.
  - edge colour follows head-node type with low alpha for density.
  - inset shows the ego-graph of the largest-degree node (after
    trimming), with labels in the `name||EntityType` format from the
    reference figure.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle as MplCircle
import networkx as nx


# Shared palette across runs so entity types keep the same colour.
PALETTE = {
    "Protein":            "#1f77b4",
    "Drug":               "#d62728",
    "Gene":               "#2ca02c",
    "Disease":            "#ff7f0e",
    "MolecularEntity":    "#9467bd",
    "BiologicalProcess":  "#8c564b",
    "CellLine":           "#e377c2",
    "Complex":            "#7f7f7f",
    "CellType":           "#bcbd22",
    "ClinicalEndpoint":   "#17becf",
    "Biomarker":          "#aec7e8",
    "TissueRegion":       "#ffbb78",
    "RNA":                "#98df8a",
    "Pathway":            "#ff9896",
    "Algorithm":          "#c5b0d5",
    "StainingMethod":     "#c49c94",
    "":                   "#b3b3b3",
}
DEFAULT_COLOR = "#aaaaaa"
BG_COLOR = "#f7f5ee"


def _node_type(G, n):
    return G.nodes[n].get("node_type", "") or ""


def trim_super_hubs(G: nx.Graph, max_deg: int) -> nx.Graph:
    """Cap each node's degree to max_deg for layout purposes only.

    A hub with deg>500 makes spring_layout explode into a starburst
    that hides the rest of the structure. We keep the top-max_deg
    neighbours (by their own degree) and drop the edges beyond that.
    """
    H = G.copy()
    for n in list(H.nodes()):
        if H.degree(n) > max_deg:
            nbrs = sorted(H.neighbors(n), key=lambda x: -H.degree(x))
            for v in nbrs[max_deg:]:
                if H.has_edge(n, v):
                    H.remove_edge(n, v)
    H.remove_nodes_from([n for n, d in H.degree() if d == 0])
    return H


def render(
    graphml_path: Path,
    out_path: Path,
    title: str,
    hub_cap: int = 60,
    inset_neighbours: int = 50,
    inset_labels: int = 30,
) -> None:
    G_dir = nx.read_graphml(graphml_path)
    U = nx.Graph(G_dir)
    print(f"[{graphml_path.name}] raw: {U.number_of_nodes()} nodes  "
          f"{U.number_of_edges()} undirected edges")

    H = trim_super_hubs(U, max_deg=hub_cap)
    print(f"[{graphml_path.name}] trim(hub_cap={hub_cap}): {H.number_of_nodes()} nodes  "
          f"{H.number_of_edges()} edges")

    # type histogram on the trimmed graph (drives the legend)
    type_counts = Counter(_node_type(H, n) for n in H.nodes())

    # dense circular hairball
    pos = nx.spring_layout(H, seed=42, k=0.32, iterations=120)

    # Smaller figure → same nodes/edges occupy a tighter canvas, graph
    # looks denser visually without touching layout / hub-cap settings.
    fig = plt.figure(figsize=(11, 8), facecolor=BG_COLOR)
    ax_legend = fig.add_axes([0.01, 0.02, 0.14, 0.96], facecolor=BG_COLOR)
    ax_main = fig.add_axes([0.15, 0.02, 0.84, 0.96], facecolor=BG_COLOR)
    for ax in (ax_legend, ax_main):
        ax.axis("off")

    ax_legend.text(0.02, 0.995, title, transform=ax_legend.transAxes,
                   fontsize=12, fontweight="bold", va="top")

    legend_handles = [
        Line2D([0], [0], marker="s", linestyle="", markersize=11,
               markerfacecolor=PALETTE.get(t, DEFAULT_COLOR), markeredgecolor="none",
               label=(t if t else "(unknown)") + f"  ({c})")
        for t, c in sorted(type_counts.items(), key=lambda kv: -kv[1])
    ]
    ax_legend.legend(handles=legend_handles, loc="upper left",
                     bbox_to_anchor=(0.0, 0.96), frameon=False, fontsize=8.5,
                     handletextpad=0.6, labelspacing=0.85)

    # main hairball
    node_colors = [PALETTE.get(_node_type(H, n), DEFAULT_COLOR) for n in H.nodes()]
    node_sizes = [6 + 1.2 * H.degree(n) for n in H.nodes()]
    edge_colors = [PALETTE.get(_node_type(H, u), DEFAULT_COLOR) for u, v in H.edges()]

    nx.draw_networkx_edges(H, pos, alpha=0.12, width=0.35,
                           edge_color=edge_colors, ax=ax_main)
    nx.draw_networkx_nodes(H, pos, node_size=node_sizes, node_color=node_colors,
                           linewidths=0, alpha=0.92, ax=ax_main)

    # tighten main-axes limits to the actual layout extent so the content
    # fills the panel instead of sitting inside a generous margin
    xs_main = [p[0] for p in pos.values()]
    ys_main = [p[1] for p in pos.values()]
    pad = 0.03 * max(max(xs_main) - min(xs_main), max(ys_main) - min(ys_main))
    ax_main.set_xlim(min(xs_main) - pad, max(xs_main) + pad)
    ax_main.set_ylim(min(ys_main) - pad, max(ys_main) + pad)

    # zoomed-in inset around the biggest hub
    top_hub = max(H.degree(), key=lambda x: x[1])[0]
    hub_x, hub_y = pos[top_hub]
    nbrs = sorted(H.neighbors(top_hub), key=lambda n: -H.degree(n))[:inset_neighbours]
    sub = H.subgraph([top_hub] + nbrs).copy()

    # red ring on the main graph marks the zoom source
    margin = 0.07
    ax_main.add_patch(MplCircle((hub_x, hub_y), margin, fill=False,
                                 edgecolor="red", lw=2.0, zorder=5))

    # inset axes (bottom right), bordered by a red circle to echo the marker
    ax_inset = fig.add_axes([0.62, 0.02, 0.37, 0.37], facecolor=BG_COLOR, zorder=6)
    ax_inset.axis("off")
    ax_inset.set_xlim(-1.15, 1.15)
    ax_inset.set_ylim(-1.15, 1.15)
    ax_inset.set_aspect("equal")
    ax_inset.add_patch(MplCircle((0, 0), 1.08, fill=False, edgecolor="red", lw=2.0))

    sub_pos = nx.spring_layout(sub, seed=3, k=0.9, iterations=120)
    # scale to fit inset unit circle
    xs = [p[0] for p in sub_pos.values()]
    ys = [p[1] for p in sub_pos.values()]
    cx = (max(xs) + min(xs)) / 2
    cy = (max(ys) + min(ys)) / 2
    span = max(max(xs) - min(xs), max(ys) - min(ys)) or 1.0
    scaled = {n: ((px - cx) / span * 1.8, (py - cy) / span * 1.8)
              for n, (px, py) in sub_pos.items()}

    sub_edge_colors = [PALETTE.get(_node_type(sub, u), DEFAULT_COLOR)
                       for u, v in sub.edges()]
    sub_node_colors = [PALETTE.get(_node_type(sub, n), DEFAULT_COLOR)
                       for n in sub.nodes()]
    sub_sizes = [120 if n == top_hub else 30 for n in sub.nodes()]

    nx.draw_networkx_edges(sub, scaled, alpha=0.55, width=0.6,
                           edge_color=sub_edge_colors, ax=ax_inset)
    nx.draw_networkx_nodes(sub, scaled, node_size=sub_sizes, node_color=sub_node_colors,
                           linewidths=0.4, edgecolors="#222", alpha=0.95, ax=ax_inset)

    def _label(n: str) -> str:
        t = _node_type(sub, n) or "?"
        short = n if len(n) <= 40 else n[:38] + "…"
        return f"{short}||{t}"

    labeled = sorted(sub.nodes(), key=lambda n: -sub.degree(n))[:inset_labels]
    nx.draw_networkx_labels(sub, scaled, labels={n: _label(n) for n in labeled},
                            font_size=5.6, ax=ax_inset)
    ax_inset.text(0.5, -0.04, "Zoomed-in View", transform=ax_inset.transAxes,
                  ha="center", fontsize=11, style="italic")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--graphml", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--title", default="")
    parser.add_argument("--hub-cap", type=int, default=60,
                        help="Cap per-node degree for layout (prevents super-hub starburst)")
    parser.add_argument("--inset-neighbours", type=int, default=50)
    parser.add_argument("--inset-labels", type=int, default=30)
    args = parser.parse_args()

    title = args.title or args.graphml.parent.name
    render(args.graphml, args.output, title,
           hub_cap=args.hub_cap,
           inset_neighbours=args.inset_neighbours,
           inset_labels=args.inset_labels)


if __name__ == "__main__":
    main()
