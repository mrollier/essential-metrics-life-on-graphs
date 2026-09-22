"""
Tests of the numpy LLNA engine (llna_numpy.py) and the network constructors used
by the synchronisation-task experiment.

Run with the `essential-metrics` conda environment:
    python -m pytest -p no:cacheprovider
"""
import numpy as np
import pytest

import llna_numpy as ln
import networks as nw
from rules import binary_indices, return_equivalent_rule

RESOLUTION = 9
ALL = (1 << RESOLUTION) - 1  # 511

# The 27 resolution-9 candidate rules for the synchronisation task at k = 8
# (self-equivalent, mean-field curve >= 1 - rho for 0 < rho < 1/2, (511, 0) excluded).
CANDIDATES_R9_K8 = [
    (15, 31), (23, 47), (31, 15), (47, 23), (63, 7), (79, 27), (95, 11), (127, 3),
    (143, 29), (159, 13), (175, 21), (191, 5), (223, 9), (255, 1), (271, 30), (279, 46),
    (287, 14), (303, 22), (319, 6), (335, 26), (351, 10), (383, 2), (399, 28), (415, 12),
    (431, 20), (447, 4), (479, 8),
]


#%% helpers

def _reverse_bits(n:np.ndarray) -> np.ndarray:
    """Mirror the RESOLUTION-bit masks in n (interval k -> RESOLUTION - 1 - k)."""
    out = np.zeros_like(n)
    for k in range(RESOLUTION):
        out |= ((n >> k) & 1) << (RESOLUTION - 1 - k)
    return out


def self_equivalent_rules() -> set:
    """All (beta, sigma) pairs of resolution 9 that equal their own equivalent rule."""
    beta = np.arange(1 << RESOLUTION)[:, None]
    sigma = np.arange(1 << RESOLUTION)[None, :]
    # equivalent rule: B_equiv = complement(mirror(S)), S_equiv = complement(mirror(B))
    mask = (beta == (~_reverse_bits(sigma) & ALL)) & (sigma == (~_reverse_bits(beta) & ALL))
    b, s = np.nonzero(mask)
    return set(zip(b.tolist(), s.tolist()))


def candidate_rules(degree:int=8, num_points:int=101, tol:float=1e-9) -> set:
    """The synchronisation-task candidate filter, using the package's mean-field curve."""
    analysis = pytest.importorskip("analysis")
    rho = np.linspace(0, 1, num_points)[1:-1]
    lower = rho < 0.5
    out = set()
    for beta, sigma in self_equivalent_rules():
        if (beta, sigma) == (ALL, 0):
            continue  # pure complementation: mean-field curve identically 1 - rho
        f = analysis.mean_field_dens_propagation(
            RESOLUTION, binary_indices(beta), binary_indices(sigma), rho, degree, iso=True)
        if np.all(f[lower] >= 1 - rho[lower] - tol):
            out.add((beta, sigma))
    return out


#%% tests

def test_interval_lut():
    """The integer look-up table agrees with the package's float interval encodings."""
    max_deg = 24
    lut = ln.interval_lut(RESOLUTION, max_deg)
    references = {}
    try:
        import analysis
        references["analysis._interval_encoding"] = lambda rhos: (
            analysis._interval_encoding(RESOLUTION, rhos[np.newaxis, :], iso=True)[0].argmax(axis=1))
    except ImportError:
        pass
    try:
        import torch as tc
        from automata import LLNA
        model = LLNA(RESOLUTION, [0], [0], iso=True)
        references["LLNA.interval_encoding"] = lambda rhos: (
            model.interval_encoding(tc.tensor(rhos[np.newaxis, :]))[0].argmax(dim=1).numpy())
    except ImportError:
        pass
    if not references:
        pytest.skip("neither analysis nor automata (torch) is importable")
    mismatches = []
    for deg in range(1, max_deg + 1):
        q = np.arange(deg + 1)
        for name, encode in references.items():
            ref = encode(q / deg)
            for qq in q:
                if lut[deg, qq] != ref[qq]:
                    mismatches.append((name, deg, int(qq), int(lut[deg, qq]), int(ref[qq])))
    if mismatches:
        boundary = [m for m in mismatches if m[1] % RESOLUTION == 0]
        print("interval mismatches (reference, degree, q, numpy, reference value):")
        for m in mismatches:
            print("  ", m)
        print(f"{len(boundary)} of {len(mismatches)} mismatches are at degrees that are a "
              f"multiple of {RESOLUTION} (exact boundary densities)")
    assert not mismatches, f"{len(mismatches)} interval mismatches, see test output"


def test_self_equivalence_and_complementation():
    """(23, 47) is self-equivalent, there are 512 such rules, and (B, S) -> (B^C, S^C)
    preserves self-equivalence; the vectorised enumeration agrees with rules.return_equivalent_rule."""
    se = self_equivalent_rules()
    assert (23, 47) in se
    assert len(se) == 512
    assert all((ALL - b, ALL - s) in se for b, s in se)
    # cross-check against the package's (slower) implementation on a sample of pairs
    rng = np.random.default_rng(0)
    sample = [(int(b), int(s)) for b, s in rng.integers(0, ALL + 1, size=(300, 2))] + list(se)[:100]
    for b, s in sample:
        equiv = return_equivalent_rule(RESOLUTION, binary_indices(b), binary_indices(s),
                                       return_decimals=True)
        assert ((b, s) in se) == (equiv == (b, s)), (b, s, equiv)


def test_candidate_filter():
    """The filter returns exactly the 27 candidates, whose complements are self-equivalent too."""
    cands = candidate_rules()
    assert cands == set(CANDIDATES_R9_K8)
    se = self_equivalent_rules()
    complemented = {(ALL - b, ALL - s) for b, s in cands}
    assert len(complemented) == 27 and complemented <= se


def test_central_interval_unreachable_at_k7():
    """No density q/7 lies in the central interval R_4 = [4/9, 5/9]; for k = 4 only the
    intervals {0, 2, 4, 6, 8} are reachable."""
    lut = ln.interval_lut(RESOLUTION, 8)
    assert 4 not in lut[7, :8]
    assert set(lut[4, :5].tolist()) == {0, 2, 4, 6, 8}


def test_homogeneous_alternation():
    """Under (23, 47) the all-0 configuration maps to all-1 and back."""
    rule = ln.Rule(RESOLUTION, 23, 47)
    A = ln.adjacency(nw.torus_lattice(6, 8))
    zeros = np.zeros((36, 1), dtype=np.int32)
    ones = ln.step(A, rule, zeros)
    assert np.all(ones == 1)
    assert np.all(ln.step(A, rule, ones) == 0)
    t_hom = ln.run_until_homogeneous(A, rule, np.concatenate([zeros, ones], axis=1), 5)
    assert t_hom.tolist() == [0, 0]


def test_torus_lattice_matches_create_2d_torus_lattice():
    """torus_lattice reproduces create_2d_torus_lattice for k = 4 and 8 and is k-regular."""
    for k in (4, 8, 12):
        g = nw.torus_lattice(10, k)
        assert set(g.degree()) == {k}
        if k in (4, 8):
            old = nw.create_2d_torus_lattice(10, k)
            assert set(g.get_edgelist()) == {tuple(sorted(e)) for e in old.get_edgelist()}


def test_ws_ring_seed_is_reproducible():
    """The same seed gives the same ring; an odd k gives degree k - 1 before rewiring."""
    g1 = nw.ws_ring(100, 8, 0.2, seed=3)
    g2 = nw.ws_ring(100, 8, 0.2, seed=3)
    assert g1.get_edgelist() == g2.get_edgelist()
    assert set(nw.ws_ring(100, 7, 0.0, seed=0).degree()) == {6}
