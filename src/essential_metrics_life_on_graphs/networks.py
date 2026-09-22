import igraph as ig
import random
import numpy as np

def create_2d_torus_lattice(L, degree=8):
    """Create a 2D torus lattice with degree 8 or 4."""
    g = ig.Graph()
    n = L * L
    g.add_vertices(n)
    
    edges = []
    for y in range(L):
        for x in range(L):
            i = _node_index(x, y, L)
            if degree == 8:
                # 8 neighbors (Moore neighborhood: cardinal + diagonal)
                neighbors = [
                    (x+1, y), (x-1, y), (x, y+1), (x, y-1),
                    (x+1, y+1), (x-1, y-1), (x+1, y-1), (x-1, y+1)
                ]
            elif degree == 4:
                # 4 neighbors (von Neumann neighborhood: cardinal only)
                neighbors = [
                    (x+1, y), (x-1, y), (x, y+1), (x, y-1)
                ]
            else:
                raise ValueError("Degree must be either 4 (von Neumann) or 8 (Moore).")
            for nx, ny in neighbors:
                j = _node_index(nx, ny, L)
                if i < j:  # avoid double-adding edges
                    edges.append((i, j))
    g.add_edges(edges)
    return g

def watts_strogatz_rewire(g, p):
    """Rewire edges in a copy of graph g with probability p (Watts-Strogatz style)."""
    if not g.is_connected():
        raise ValueError("Input graph must be connected.")
    try_counter = 0
    max_counter = 10
    while True:
        try_counter += 1
        g_copy = g.copy()  # Make a copy of the original graph
        n = len(g_copy.vs)  # Number of vertices in the graph
        edges_to_rewire = list(g_copy.get_edgelist())  # Get the list of all edges
        for edge in edges_to_rewire:
            if random.random() < p:  # With probability p, rewire the edge
                source, target = edge
                g_copy.delete_edges([edge])  # Remove the current edge
                
                # Avoid self-loops and existing edges
                possible_targets = set(range(n)) - {source} - set(g_copy.neighbors(source))
                if possible_targets:
                    new_target = random.choice(list(possible_targets))  # Choose a new target randomly
                    g_copy.add_edges([(source, new_target)])  # Add the new edge
                else:
                    raise ValueError(f"No possible target for moving the edge from node {source}.")
        if g_copy.is_connected():
            return g_copy
        if try_counter == max_counter:
            raise ValueError(f"It was not possible to construct a connected Watts-Strogatz graph.")

#%% helper functions

def _node_index(x, y, L):
    """Convert 2D coordinates to node index in 1D."""
    return (y % L) * L + (x % L)

def torus_lattice(L, k=8):
    """
    Create a toroidal L x L lattice with a von Neumann, Moore or radius-2 von Neumann neighbourhood.

    This generalises `create_2d_torus_lattice` (which is kept unchanged) to k = 12 and builds
    the edge set once, so that each unordered pair of nodes appears exactly once.

    Parameters
    ----------
    L : int
        Side length of the lattice; the network has N = L * L nodes.
    k : int, optional
        Degree of every node: 4 (von Neumann), 8 (Moore) or 12 (radius-2 von Neumann, i.e. all
        offsets with 0 < |dx| + |dy| <= 2). Default 8.

    Returns
    -------
    g : igraph.Graph
        The k-regular toroidal lattice.
    """
    if k == 4:
        offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    elif k == 8:
        offsets = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]
    elif k == 12:
        offsets = [(dx, dy) for dx in range(-2, 3) for dy in range(-2, 3)
                   if 0 < abs(dx) + abs(dy) <= 2]
    else:
        raise ValueError("k must be 4 (von Neumann), 8 (Moore) or 12 (radius-2 von Neumann).")
    edges = set()
    for y in range(L):
        for x in range(L):
            i = _node_index(x, y, L)
            for dx, dy in offsets:
                j = _node_index(x + dx, y + dy, L)
                if i < j:
                    edges.add((i, j))
    return ig.Graph(n=L * L, edges=sorted(edges))


def ws_rewire_grid(g, p, rng, max_tries=10):
    """
    Rewire a copy of network g with probability p per edge (Watts-Strogatz style), using a
    numpy random number generator.

    This follows `watts_strogatz_rewire` (which is kept unchanged): every edge is visited once;
    with probability p it is detached from its second endpoint and the first endpoint is
    reattached to a uniformly random node (no self-loops, no duplicate edges). Disconnected
    results are rejected and the rewiring is repeated.

    Parameters
    ----------
    g : igraph.Graph
        Connected undirected network.
    p : float
        Rewiring probability.
    rng : numpy.random.Generator
        Random number generator; use `numpy.random.default_rng(seed)` for reproducibility.
    max_tries : int, optional
        Number of attempts to obtain a connected network (default 10).

    Returns
    -------
    h : igraph.Graph
        The rewired, connected network.
    """
    n = g.vcount()
    for _ in range(max_tries):
        adj = [set(nb) for nb in g.get_adjlist()]
        for (s, t) in g.get_edgelist():
            if rng.random() < p:
                adj[s].discard(t); adj[t].discard(s)
                candidates = np.setdiff1d(np.arange(n), np.fromiter(adj[s] | {s}, dtype=int),
                                          assume_unique=True)
                if candidates.size == 0:
                    raise ValueError(f"No possible target for moving the edge from node {s}.")
                u = int(rng.choice(candidates))
                adj[s].add(u); adj[u].add(s)
        edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
        h = ig.Graph(n=n, edges=edges)
        if h.is_connected():
            return h
    raise ValueError("It was not possible to construct a connected rewired network.")


def ws_ring(N, k, p, max_tries=500, seed=None):
    """
    Watts-Strogatz ring lattice as generated by igraph, rejecting disconnected results.

    This is the network family behind the published synchronisation figure (see
    `data/fssp/paper-2025/README.md`): `igraph.Graph.Watts_Strogatz(dim=1, size=N, nei=k//2, p=p)`.
    Note that the actual degree is 2 * (k // 2), so odd k gives degree k - 1.

    Parameters
    ----------
    N : int
        Number of nodes.
    k : int
        Requested mean degree; igraph connects every node to its k // 2 nearest neighbours on
        either side.
    p : float
        Rewiring probability.
    max_tries : int, optional
        Number of attempts to obtain a connected network (default 500).
    seed : int, optional
        If given, Python's `random` module (the generator igraph draws from) is seeded before
        generating. This also affects `watts_strogatz_rewire`, which uses the same generator.

    Returns
    -------
    g : igraph.Graph
        The connected ring lattice.
    """
    if seed is not None:
        random.seed(seed)
    for _ in range(max_tries):
        g = ig.Graph.Watts_Strogatz(dim=1, size=N, nei=k // 2, p=p)
        if g.is_connected():
            return g
    raise ValueError("It was not possible to construct a connected Watts-Strogatz ring.")
