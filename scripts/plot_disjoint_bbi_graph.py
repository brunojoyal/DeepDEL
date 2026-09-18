#!/usr/bin/env python3
"""plot_disjoint_bbi_graph.py

Build and render a graph from a ranked_list.csv.

Vertices
--------
Each vertex is a unique value from the chosen pool column `bb{i}_pool` among
rows with reward > threshold.

Edges
-----
Two vertices are connected by an undirected edge iff their pipe-separated ID
sets are disjoint.

Example
-------
python scripts/plot_disjoint_bbi_graph.py \
  --csv data/scored_libraries/not_random/3.3.3/ranked_list.csv \
  --i 1 --threshold 26 --out disjoint_bb1_graph.png
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from typing import Dict, FrozenSet, Iterable, Tuple

import pandas as pd

import matplotlib

# Non-interactive backend: this is typically run on servers / via CLI.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot a graph where vertices are bb{i}_pool values (reward>threshold) and edges connect disjoint pools."
    )
    p.add_argument("--csv", required=True, help="Path to ranked_list.csv")
    p.add_argument("--i", required=True, type=int, choices=(1, 2, 3), help="Which pool column to use: bb1_pool/bb2_pool/bb3_pool")
    p.add_argument(
        "--threshold",
        type=float,
        default=26.0,
        help="Only include libraries with reward > threshold (default: 26)",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output PNG path. Default: disjoint_bb{i}_graph_thr{threshold}.png",
    )
    p.add_argument(
        "--max-vertices",
        type=int,
        default=None,
        help="Optional cap on number of vertices (keeps highest-reward unique pools). Useful if graph is huge.",
    )
    p.add_argument(
        "--layout",
        choices=("spring", "kamada_kawai"),
        default="spring",
        help="Graph layout algorithm.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for layout reproducibility (where applicable).",
    )
    p.add_argument(
        "--node-size",
        type=float,
        default=30.0,
        help="Node size for drawing (default tuned for 1000s of nodes).",
    )
    p.add_argument(
        "--node-color",
        choices=("degree", "constant"),
        default="degree",
        help="How to color nodes. 'degree' is recommended for large graphs.",
    )
    p.add_argument(
        "--cmap",
        default="viridis",
        help="Matplotlib colormap name used when --node-color=degree.",
    )
    p.add_argument(
        "--degree-log-color",
        action="store_true",
        help="Color nodes by log1p(degree) instead of degree (helps when hubs dominate).",
    )
    p.add_argument(
        "--no-edges",
        action="store_true",
        help="Do not draw edges (useful when graph is very dense / hairball).",
    )
    p.add_argument(
        "--max-edges-draw",
        type=int,
        default=200_000,
        help="Max number of edges to draw (randomly sampled). Helps performance for dense graphs. Use 0 to disable sampling.",
    )
    p.add_argument(
        "--edge-alpha",
        type=float,
        default=0.08,
        help="Edge alpha (default tuned for 1000s of nodes).",
    )
    p.add_argument(
        "--edge-width",
        type=float,
        default=0.5,
        help="Edge width (default tuned for 1000s of nodes).",
    )
    p.add_argument(
        "--with-labels",
        action="store_true",
        help="Draw node labels (can be cluttered for large graphs).",
    )
    return p.parse_args()


def parse_pool_to_set(pool: str) -> FrozenSet[int]:
    """Parse a pipe-separated pool string like '1633|1675|3757' -> frozenset({1633,1675,3757})."""
    if pool is None or (isinstance(pool, float) and pd.isna(pool)):
        return frozenset()
    s = str(pool).strip()
    if not s:
        return frozenset()
    out = []
    for tok in s.split("|"):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return frozenset(out)


def load_pools(
    csv_path: str,
    i: int,
    threshold: float,
    max_vertices: int | None,
) -> Dict[str, FrozenSet[int]]:
    pool_col = f"bb{i}_pool"

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV '{csv_path}': {e}")

    required = {"reward", pool_col}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns in CSV: {missing}. Available: {list(df.columns)}")

    df["reward"] = pd.to_numeric(df["reward"], errors="coerce")
    df = df.dropna(subset=["reward", pool_col])
    df = df[df["reward"] > float(threshold)]

    if df.empty:
        return {}

    # For each unique pool string, keep the max reward seen (helps sorting if max_vertices is set)
    best_reward = df.groupby(pool_col, sort=False)["reward"].max().sort_values(ascending=False)

    if max_vertices is not None:
        best_reward = best_reward.head(int(max_vertices))

    pools = {pool_str: parse_pool_to_set(pool_str) for pool_str in best_reward.index.tolist()}
    return pools


def build_disjoint_graph(pools: Dict[str, FrozenSet[int]]) -> nx.Graph:
    g = nx.Graph()
    for node, s in pools.items():
        g.add_node(node, size=len(s))

    items = list(pools.items())
    for (n1, s1), (n2, s2) in combinations(items, 2):
        if s1.isdisjoint(s2):
            g.add_edge(n1, n2)
    return g


def sample_edges_for_drawing(g: nx.Graph, max_edges: int, seed: int) -> nx.Graph:
    """Return a new graph with the same nodes but only up to max_edges edges.

    This is purely for visualization: drawing millions of edges is slow and
    visually indistinguishable from a hairball.
    """
    if max_edges is None or max_edges <= 0:
        return g
    if g.number_of_edges() <= max_edges:
        return g

    import random

    rng = random.Random(seed)
    edges = list(g.edges())
    rng.shuffle(edges)
    keep = edges[: int(max_edges)]

    h = nx.Graph()
    h.add_nodes_from(g.nodes(data=True))
    h.add_edges_from(keep)
    return h


def compute_layout(g: nx.Graph, layout: str, seed: int):
    if g.number_of_nodes() == 0:
        return {}
    if layout == "kamada_kawai":
        return nx.kamada_kawai_layout(g)
    # spring
    return nx.spring_layout(g, seed=seed)


def main() -> int:
    args = parse_args()

    out = args.out
    if out is None:
        thr_str = (f"{args.threshold}".rstrip("0").rstrip(".") if isinstance(args.threshold, float) else str(args.threshold))
        out = f"disjoint_bb{args.i}_graph_thr{thr_str}.png"

    try:
        pools = load_pools(args.csv, args.i, args.threshold, args.max_vertices)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    g = build_disjoint_graph(pools)
    print(
        f"Loaded {len(pools)} unique bb{args.i}_pool vertices with reward>{args.threshold}. "
        f"Graph: |V|={g.number_of_nodes()} |E|={g.number_of_edges()}",
        file=sys.stderr,
    )

    if g.number_of_nodes() == 0:
        print("Nothing to plot (no vertices).", file=sys.stderr)
        return 0

    pos = compute_layout(g, args.layout, args.seed)

    deg = dict(g.degree())

    # Node colors
    node_colors = None
    add_colorbar = False
    if args.node_color == "degree":
        values = [deg[n] for n in g.nodes]
        if args.degree_log_color:
            values = [__import__("math").log1p(v) for v in values]
        node_colors = values
        add_colorbar = True

    # Optionally sample edges for drawing (keeps plot readable + faster)
    g_draw = g
    if not args.no_edges:
        g_draw = sample_edges_for_drawing(g, args.max_edges_draw, args.seed)

    # Draw
    fig, ax = plt.subplots(figsize=(12, 9))
    node_sizes = [args.node_size for _ in g.nodes]

    if not args.no_edges:
        nx.draw_networkx_edges(g_draw, pos, alpha=float(args.edge_alpha), width=float(args.edge_width), ax=ax)

    nodes = nx.draw_networkx_nodes(
        g,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.get_cmap(args.cmap) if node_colors is not None else None,
        linewidths=0.0,
        ax=ax,
    )

    if add_colorbar and node_colors is not None:
        # Colorbar for degree (or log-degree)
        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(args.cmap), norm=nodes.norm)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("log1p(degree)" if args.degree_log_color else "degree")

    if args.with_labels:
        nx.draw_networkx_labels(g, pos, font_size=7, ax=ax)

    ax.set_title(f"Disjointness graph: bb{args.i}_pool (reward>{args.threshold})")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)

    print(f"Saved graph PNG to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
