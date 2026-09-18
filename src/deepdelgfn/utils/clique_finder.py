from __future__ import annotations
from collections import defaultdict
from itertools import combinations
from math import ceil
from typing import Dict, Iterable, Iterator, Tuple, Set, List, Optional, TextIO

import matplotlib.pyplot as plt

Edge = Tuple[int, int]

class IncrementalWeightedGraph:
    """
    Incrementally constructs a weighted complete graph over positive integers.
    Internally stores only nonzero edges in a sparse dict keyed by (min(u,v), max(u,v)).
    """

    def __init__(self) -> None:
        self._w: Dict[Edge, float] = defaultdict(float)
        self._nodes: Set[int] = set()

    @staticmethod
    def _edge_key(u: int, v: int) -> Edge:
        if u == v:
            raise ValueError("Self-loops are not allowed (edge requires two distinct nodes).")
        if u <= 0 or v <= 0:
            raise ValueError("Nodes must be positive integers.")
        return (u, v) if u < v else (v, u)

    def update(self, B: Iterable[int], r: float) -> None:
        """Add r to the weight of each edge between distinct elements of B."""
        nodes = sorted(set(B))
        for x in nodes:
            if x < 0:
                raise ValueError("All nodes in B must be non-negative integers.")
        self._nodes.update(nodes)
        for u, v in combinations(nodes, 2):
            self._w[(u, v)] += float(r)

    def get_weight(self, u: int, v: int) -> float:
        """Return the current weight of edge {u, v}. (0.0 if unseen.)"""
        if u == v:
            return 0.0
        return self._w.get(self._edge_key(u, v), 0.0)

    def nodes(self) -> Set[int]:
        """Return the set of nodes that have appeared in any update."""
        return set(self._nodes)

    def edges(self) -> Iterator[Tuple[int, int, float]]:
        """Iterate over stored edges (u, v, w) with w != 0.0."""
        for (u, v), w in self._w.items():
            if w != 0.0:
                yield (u, v, w)

    def neighbors(self, u: int) -> Iterator[Tuple[int, float]]:
        """Iterate over (v, weight(u,v)) for neighbors v with nonzero stored edges."""
        if u <= 0:
            raise ValueError("Node must be a positive integer.")
        for (a, b), w in self._w.items():
            if a == u:
                yield (b, w)
            elif b == u:
                yield (a, w)

    def to_dense_adjacency(self, subset: Iterable[int]) -> Tuple[List[int], List[List[float]]]:
        """Build a dense adjacency matrix for a finite subset of nodes."""
        nodes = sorted(set(subset))
        idx = {n: i for i, n in enumerate(nodes)}
        n = len(nodes)
        M = [[0.0] * n for _ in range(n)]
        for (u, v), w in self._w.items():
            if u in idx and v in idx:
                i, j = idx[u], idx[v]
                M[i][j] = w
                M[j][i] = w
        return nodes, M

    def display(
        self,
        limit: Optional[int] = None,
        file: Optional[TextIO] = None,
        fmt: str = "{u} -- {v} : {w}"
    ) -> List[Tuple[int, int, float]]:
        """
        Output nonzero edges sorted by descending weight (ties broken by u,v).
        Returns the full sorted list (not just the printed subset).
        """
        edges = [(u, v, w) for (u, v), w in self._w.items() if w != 0.0]
        edges.sort(key=lambda t: (-t[2], t[0], t[1]))  # desc by weight, then (u,v)

        if limit is not None:
            edges_to_show = edges[:limit]
        else:
            edges_to_show = edges

        lines = [fmt.format(u=u, v=v, w=w) for (u, v, w) in edges_to_show]
        out = "\n".join(lines)
        if file is None:
            print(out)
        else:
            file.write(out + ("\n" if out else ""))

        return edges

    # ---------------------------
    # Clique search (threshold)
    # ---------------------------

    def _build_thresholded_adj(
        self, threshold: float, inclusive: bool
    ) -> Dict[int, Set[int]]:
        """
        Build adjacency among nodes connected by edges satisfying:
            weight > threshold   (inclusive=False)
        or  weight >= threshold (inclusive=True).
        Only uses stored edges.
        """
        adj: Dict[int, Set[int]] = defaultdict(set)
        cond = (lambda w: w >= threshold) if inclusive else (lambda w: w > threshold)
        for (u, v), w in self._w.items():
            if cond(w):
                adj[u].add(v)
                adj[v].add(u)
        return adj
    
    def search_for_cliques(
        self,
        n: int,
        threshold: float,
        inclusive: bool = True,
        limit: Optional[int] = None,
    ) -> List[Tuple[Tuple[int, ...], float]]:
        """
        Find all cliques of EXACT size n in the thresholded graph where every edge satisfies:
            weight >= threshold   if inclusive=True
            weight >  threshold   if inclusive=False

        Returns
        -------
        List[ (tuple_of_nodes, avg_weight) ]
            Each clique is a sorted tuple of node ids (ascending) and the average of all
            pairwise edge weights within the clique.

        Notes
        -----
        - Only stored (nonzero) edges are used to build the thresholded adjacency.
        Because of the threshold, any edge used in a clique is guaranteed to exist in storage.
        - Set `limit` to cap the number of cliques returned.
        """
        if n <= 0:
            return []
        if n == 1:
            # Average over zero edges is undefined; return 0.0 by convention.
            return [((v,), 0.0) for v in sorted(self._nodes)]

        # Build thresholded adjacency
        adj = self._build_thresholded_adj(threshold, inclusive)

        # Pre-prune: nodes must have degree >= n-1
        candidates = [v for v, nbrs in adj.items() if len(nbrs) >= n - 1]
        if len(candidates) < n:
            return []

        cand_set = set(candidates)
        nbrs: Dict[int, Set[int]] = {v: (adj[v] & cand_set) for v in candidates}

        # Helper to get stored weight (edges in thresholded graph must exist)
        def w(u: int, v: int) -> float:
            key = (u, v) if u < v else (v, u)
            return self._w.get(key, 0.0)

        # Precompute denominator for the clique-average
        denom = n * (n - 1) // 2

        P = set(sorted(candidates))
        X: Set[int] = set()
        R: Tuple[int, ...] = tuple()
        results: List[Tuple[Tuple[int, ...], float]] = []

        def backtrack(R: Tuple[int, ...], P: Set[int], X: Set[int]) -> None:
            nonlocal results, limit
            if len(R) == n:
                clique = tuple(sorted(R))
                # Sum all pairwise weights within the clique
                total = 0.0
                for i in range(n):
                    ui = clique[i]
                    for j in range(i + 1, n):
                        vj = clique[j]
                        total += w(ui, vj)
                avg_w = total / denom if denom > 0 else 0.0
                results.append((clique, avg_w))
                return

            # Bound: impossible to reach size n
            if len(R) + len(P) < n:
                return
            if limit is not None and len(results) >= limit:
                return

            # Pivoting: choose u in P ∪ X maximizing |P ∩ N(u)|
            union = P | X
            if union:
                u = max(union, key=lambda x: len(P & nbrs.get(x, set())))
                iterate_over = P - nbrs.get(u, set())
            else:
                iterate_over = set(P)

            for v in sorted(iterate_over):
                R_next = R + (v,)
                P_next = P & nbrs.get(v, set())
                X_next = X & nbrs.get(v, set())
                backtrack(R_next, P_next, X_next)
                P.remove(v)
                X.add(v)
                if limit is not None and len(results) >= limit:
                    return

            backtrack(R, P, X)
            # Deterministic output order: by clique tuple, or you can sort by avg weight if preferred
            results.sort(key=lambda item: item[0])
            return results[:limit] if limit is not None else results

    # ---------------------------
    # Clique search (global quantile)
    # ---------------------------

    @staticmethod
    def _nearest_rank_quantile(values: List[float], q: float) -> float:
        """
        Nearest-rank quantile (inclusive):
            r = ceil(q * m); return values_sorted[r-1]
        with q in [0,1]; q=0 returns min, q=1 returns max.
        """
        if not values:
            raise ValueError("No edges to compute a quantile over.")
        if not (0.0 <= q <= 1.0):
            raise ValueError("q must be in [0, 1].")
        ws = sorted(values)
        m = len(ws)
        if q == 0.0:
            return ws[0]
        r = ceil(q * m)
        return ws[r - 1]

    def search_for_cliques_by_quantile(
        self,
        n: int,
        q: float,
        inclusive: bool = True,
        limit: Optional[int] = None,
    ) -> List[Tuple[int, ...]]:
        """
        Compute a GLOBAL quantile cutoff t over ALL stored (nonzero) edge weights,
        then find all cliques of EXACT size n where every edge satisfies
            weight >= t   if inclusive=True
            weight >  t   if inclusive=False
        """
        weights = [w for w in self._w.values() if w != 0.0]
        if not weights:
            return []
        t = self._nearest_rank_quantile(weights, q)
        return self.search_for_cliques(n=n, threshold=t, inclusive=inclusive, limit=limit)

    # ---------------------------
    # Histogram of nonzero edge weights
    # ---------------------------

    def plot_weight_hist(
        self,
        bins: int | List[float] = 30,
        density: bool = False,
        log: bool = False,
        ax: Optional[plt.Axes] = None,
        show: bool = True,
        title: Optional[str] = None,
    ):
        """
        Plot a histogram of NONZERO edge weights using matplotlib.

        Parameters
        ----------
        bins : int or sequence, default 30
            Number of bins or explicit bin edges.
        density : bool, default False
            If True, normalize to form a probability density.
        log : bool, default False
            If True, set a log scale on the y-axis.
        ax : matplotlib Axes, optional
            Existing axes to draw on. If None, creates a new figure/axes.
        show : bool, default True
            If True, calls plt.show() at the end.
        title : str, optional
            Title for the plot. If None, a default title is used.

        Returns
        -------
        (fig, ax) : tuple
            The matplotlib Figure and Axes used for the plot.
        """
        weights = [w for w in self._w.values() if w != 0.0]

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        if not weights:
            ax.text(0.5, 0.5, "No nonzero edges", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel("Edge weight")
            ax.set_ylabel("Count")
            ax.set_title(title or "Histogram of nonzero edge weights")
            if show:
                plt.show()
            return fig, ax

        ax.hist(weights, bins=bins, density=density)
        ax.set_xlabel("Edge weight")
        ax.set_ylabel("Density" if density else "Count")
        if log:
            ax.set_yscale("log")
        ax.set_title(title or "Histogram of nonzero edge weights")

        if show:
            plt.show()
        return fig, ax
