"""
The numpy engine (llna_numpy.py) must produce exactly the trajectories of the torch
reference implementation automata.LLNA. Skipped when torch or torch_geometric is not
installed.
"""
import numpy as np
import pytest

tc = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

import llna_numpy as ln
import networks as nw
from automata import LLNA
from rules import binary_indices, return_life_like_dict


def test_trajectories_match_torch():
    rng = np.random.default_rng(2025)
    g = nw.ws_rewire_grid(nw.torus_lattice(30, 8), 0.2, rng)
    N = g.vcount()
    A = ln.adjacency(g)
    edges = np.array(g.get_edgelist())
    E = tc.tensor(np.concatenate([edges, edges[:, ::-1]]).T, dtype=tc.long)  # both directions
    # 10 fixed initial configurations: two at each of five densities
    S0 = np.concatenate([ln.init_configs(N, d, 2, rng) for d in (0.1, 0.3, 0.5, 0.7, 0.9)], axis=1)
    T = 50
    rules = [(23, 47), return_life_like_dict()["morley"]]
    rules += [(int(b), int(s)) for b, s in rng.integers(0, 512, size=(3, 2))]
    for beta, sigma in rules:
        H_np = ln.trajectory(A, ln.Rule(9, beta, sigma), S0, T)          # T+1 x N x M
        model = LLNA(9, binary_indices(beta), binary_indices(sigma), iso=True)
        with tc.no_grad():
            H_tc = model(E, tc.tensor(S0.T, dtype=tc.float32), T)        # M x T+1 x N
        H_tc = H_tc.numpy().transpose(1, 2, 0).astype(np.int32)
        assert np.array_equal(H_np, H_tc), f"rule ({beta}, {sigma}): {(H_np != H_tc).sum()} differing entries"
