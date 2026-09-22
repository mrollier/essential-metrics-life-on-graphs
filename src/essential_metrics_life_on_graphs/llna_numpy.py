"""
A numpy/scipy implementation of the Life-like network automaton (LLNA) for the
synchronisation-task experiments.

This module re-implements ``automata.LLNA`` with ``iso=True`` without torch, so
that large sweeps can be run on any machine. The torch class remains the
reference implementation; ``tests/test_torch_parity.py`` checks that both
produce identical trajectories.

The update rule follows ``automata.LLNA`` exactly:

* the neighbourhood density of a node is q/deg, where q is the number of living
  neighbours (own state excluded, "mean" aggregation);
* a resolution r partitions [0, 1] into r intervals

      R_k = [k/r, (k+1)/r)   for k < (r-1)/2
      R_k = [k/r, (k+1)/r]   for k = (r-1)/2   (closed central interval)
      R_k = (k/r, (k+1)/r]   for k > (r-1)/2

  membership is decided with integer arithmetic (q*r compared with k*deg), which
  is the float logic of the original code without rounding surprises;
* the next state is 1 iff (s = 0 and R_k in B) or (s = 1 and R_k in S).
"""
import numpy as np
import scipy.sparse as sp
import igraph as ig

from rules import binary_indices


#%% rule

def interval_lut(resolution:int, max_deg:int) -> np.ndarray:
    """
    Look-up table of the density interval for every (degree, number of living neighbours).

    Parameters
    ----------
    resolution : int
        Odd positive integer indicating the resolution of the local update rule.
    max_deg : int
        Largest node degree that the table must cover.

    Returns
    -------
    lut : numpy.ndarray
        Integer array of shape (max_deg + 1, max_deg + 1) such that lut[deg, q] is the
        index k of the interval R_k that contains the density q/deg (isotropic case).
        Entries with q > deg or deg = 0 are -1.
    """
    lut = np.full((max_deg + 1, max_deg + 1), -1, dtype=np.int8)
    mid = (resolution - 1) // 2
    for deg in range(1, max_deg + 1):
        for q in range(deg + 1):
            x = q * resolution  # compare x with k*deg  <=>  q/deg with k/resolution
            for k in range(resolution):
                lo, hi = k * deg, (k + 1) * deg
                if k < mid:
                    ok = lo <= x < hi
                elif k == mid:
                    ok = lo <= x <= hi
                else:
                    ok = lo < x <= hi
                if ok:
                    lut[deg, q] = k
                    break
            assert lut[deg, q] >= 0
    return lut


class Rule:
    """
    An isotropic LLNA local update rule, decoded from its beta and sigma integers.

    Parameters
    ----------
    resolution : int
        Odd positive integer indicating the resolution of the local update rule.
    beta : int
        Integer whose set bits (bit 0 = interval R_0) are the B set.
    sigma : int
        Integer whose set bits are the S set.
    max_deg : int, optional
        Largest node degree the interval look-up table must cover (default 64).

    Attributes
    ----------
    B, S : numpy.ndarray
        Boolean arrays of length `resolution`; True where the interval is in the set.
    lut : numpy.ndarray
        The interval look-up table, see `interval_lut`.
    """
    def __init__(self, resolution:int, beta:int, sigma:int, max_deg:int=64):
        if resolution % 2 == 0:
            raise ValueError("Resolution must be an odd number for the isotropic rule.")
        self.resolution, self.beta, self.sigma = resolution, beta, sigma
        self.B = np.zeros(resolution, dtype=bool); self.B[binary_indices(beta)] = True
        self.S = np.zeros(resolution, dtype=bool); self.S[binary_indices(sigma)] = True
        self.lut = interval_lut(resolution, max_deg)

    def __str__(self):
        return f"R{self.resolution}B{self.beta}S{self.sigma}"


#%% network to adjacency matrix

def adjacency(g:ig.Graph) -> sp.csr_matrix:
    """
    Sparse symmetric adjacency matrix of an undirected igraph network.

    Parameters
    ----------
    g : igraph.Graph
        Undirected network.

    Returns
    -------
    A : scipy.sparse.csr_matrix
        Adjacency matrix of shape (N, N) with integer entries.
    """
    A = np.array(g.get_edgelist())
    n = g.vcount()
    rows = np.concatenate([A[:, 0], A[:, 1]]); cols = np.concatenate([A[:, 1], A[:, 0]])
    return sp.csr_matrix((np.ones(rows.size, dtype=np.int32), (rows, cols)), shape=(n, n))


#%% simulation

def init_configs(N:int, dens:float, M:int, rng:np.random.Generator) -> np.ndarray:
    """
    Random initial configurations with a fixed number of living nodes.

    Parameters
    ----------
    N : int
        Number of nodes.
    dens : float
        Initial density; every configuration has exactly round(dens * N) ones.
    M : int
        Number of configurations.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    S0 : numpy.ndarray
        Integer array of shape (N, M), one configuration per column.
    """
    n1 = int(np.round(dens * N))
    S = np.zeros((N, M), dtype=np.int8)
    for m in range(M):
        S[rng.choice(N, n1, replace=False), m] = 1
    return S


def step(A:sp.csr_matrix, rule:Rule, S:np.ndarray, deg:np.ndarray=None) -> np.ndarray:
    """
    One synchronous update of the LLNA on every column of S.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix
        Adjacency matrix, see `adjacency`.
    rule : Rule
        The local update rule.
    S : numpy.ndarray
        Integer array of shape (N, M) with one configuration per column.
    deg : numpy.ndarray, optional
        Node degrees (row sums of A); computed if not given.

    Returns
    -------
    S_next : numpy.ndarray
        Integer array of shape (N, M) with the updated configurations.
    """
    if deg is None:
        deg = np.asarray(A.sum(axis=1)).ravel().astype(np.int64)
    q = A @ S                                   # number of living neighbours, N x M
    k = rule.lut[deg[:, None], q]               # interval index
    return np.where(S == 0, rule.B[k], rule.S[k]).astype(np.int32)


def trajectory(A:sp.csr_matrix, rule:Rule, S0:np.ndarray, T:int) -> np.ndarray:
    """
    The full trajectory of the LLNA over T steps, for every column of S0.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix
        Adjacency matrix, see `adjacency`.
    rule : Rule
        The local update rule.
    S0 : numpy.ndarray
        Integer array of shape (N, M) with one initial configuration per column.
    T : int
        Number of time steps.

    Returns
    -------
    H : numpy.ndarray
        Integer array of shape (T + 1, N, M): the initial configurations plus T steps.
    """
    deg = np.asarray(A.sum(axis=1)).ravel().astype(np.int64)
    S = S0.astype(np.int32)
    H = [S]
    for _ in range(T):
        S = step(A, rule, S, deg)
        H.append(S)
    return np.stack(H, 0)


def run_until_homogeneous(A:sp.csr_matrix, rule:Rule, S0:np.ndarray, T:int) -> np.ndarray:
    """
    Iterate the LLNA on all columns of S0 for at most T steps and record when each
    configuration becomes homogeneous.

    A column that revisits an earlier (non-homogeneous) configuration has entered a
    cycle and can never become homogeneous, because the dynamics are deterministic.
    Such columns are dropped early; this only saves time and does not change the result.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix
        Adjacency matrix, see `adjacency`.
    rule : Rule
        The local update rule.
    S0 : numpy.ndarray
        Integer array of shape (N, M) with one initial configuration per column.
    T : int
        Maximum number of time steps.

    Returns
    -------
    t_hom : numpy.ndarray
        Integer array of shape (M,): the first t <= T (t = 0 included) at which the
        column is homogeneous (all 0 or all 1), or -1 if it never is.
    """
    N, M = S0.shape
    deg = np.asarray(A.sum(axis=1)).ravel().astype(np.int64)
    S = S0.astype(np.int32)
    active = np.arange(M)
    t_hom = np.full(M, -1, dtype=np.int64)
    seen = [set() for _ in range(M)]
    for t in range(T + 1):
        col = S.sum(axis=0)
        hom = (col == 0) | (col == N)
        if hom.any():
            t_hom[active[hom]] = t
        # cycle detection on the non-homogeneous columns
        packed = np.packbits(S.astype(np.uint8), axis=0).T  # M' x ceil(N/8)
        cyc = np.zeros(S.shape[1], dtype=bool)
        for j in range(S.shape[1]):
            if hom[j]:
                continue
            h = packed[j].tobytes()
            if h in seen[active[j]]:
                cyc[j] = True
            else:
                seen[active[j]].add(h)
        drop = hom | cyc
        if drop.any():
            keep = ~drop
            S = S[:, keep]; active = active[keep]
            if active.size == 0:
                break
        if t == T:
            break
        S = step(A, rule, S, deg)
    return t_hom


def success_rate(graphs, rule:Rule, dens:float, M:int, T:int, rng:np.random.Generator) -> float:
    """
    Fraction of (network, initial configuration) samples that reach homogeneity within T steps.

    Parameters
    ----------
    graphs : list of igraph.Graph
        Network realisations.
    rule : Rule
        The local update rule.
    dens : float
        Initial density of every configuration.
    M : int
        Number of initial configurations per network.
    T : int
        Maximum number of time steps.
    rng : numpy.random.Generator
        Random number generator for the initial configurations.

    Returns
    -------
    rate : float
        Number of successful samples divided by len(graphs) * M.
    """
    ok, tot = 0, 0
    for g in graphs:
        A = adjacency(g)
        S0 = init_configs(g.vcount(), dens, M, rng)
        t_hom = run_until_homogeneous(A, rule, S0, T)
        ok += int((t_hom >= 0).sum()); tot += M
    return ok / tot
