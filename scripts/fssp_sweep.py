"""
Success rate of rule R9B23S47 on the global synchronisation task, against the initial density
rho^0 and against the rewiring probability p, for several mean degrees.

Two network families:
  wsg  toroidal L x L grids with von Neumann (k = 4), Moore (k = 8) or radius-2 von Neumann
       (k = 12) neighbourhoods, rewired with networks.ws_rewire_grid (the repository's
       watts_strogatz_rewire with a numpy random number generator);
  wsr  igraph Watts_Strogatz(dim=1, size=N, nei=k//2, p) ring lattices (k = 6, 8, 10), the
       family behind the published figure.

For each family and degree:
  * 21 equispaced rho^0 values at p = 0.2 (one set of network realisations for all rho^0);
  * 21 log-spaced p values in [0.01, 1] at rho^0 = 1/2 (new realisations for every p);
  * G network realisations x M initial configurations per point, at most T time steps.

The number of nodes is N = L * L by default. The ring family is not tied to a grid, so --N
sets it directly there (the thesis uses N = 1000); for the grid family N = L * L is the only
possibility and --N is refused if it disagrees. T defaults to 2 N with the effective N.

Results are checkpointed to <out>/<family>_k<k>_{dens,prob}.npy after every point, and a line
per point is appended to <out>/log.txt. Re-running with complete checkpoints computes nothing.

Seeds: the numpy generator of each (family, degree) sweep is seeded with <seed base> + k, so
the canonical data use 108, 112, 104 (wsg) and 208, 210, 206 (wsr). For the ring family a seed
for igraph's generator (Python's random module) is drawn from that generator for every
realisation.

Usage (canonical, about 40 minutes on one core):
    python scripts/fssp_sweep.py --family wsg
    python scripts/fssp_sweep.py --family wsr
Ring lattices at the thesis size (N = 1000, T = 2000):
    python scripts/fssp_sweep.py --family wsr --degrees 8 10 6 --N 1000 --seed 1200 \
        --out data/fssp/thesis-2026/wsr-N1000
Small test:
    python scripts/fssp_sweep.py --family wsg --degrees 8 --L 6 --graphs 2 --configs 2 --T 20 --out /tmp/x
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "essential_metrics_life_on_graphs"))

from llna_numpy import Rule, success_rate  # noqa: E402
from networks import torus_lattice, ws_rewire_grid, ws_ring  # noqa: E402

DEFAULT_DEGREES = {"wsg": [8, 12, 4], "wsr": [8, 10, 6]}
DEFAULT_SEED_BASE = {"wsg": 100, "wsr": 200}
DENS_ARRAY = np.linspace(0, 1, 21)
PROB_ARRAY = np.logspace(-2, 0, 21)
P_DENS = 0.2        # rewiring probability used in the rho^0 sweep
DENS_PROB = 0.5     # initial density used in the p sweep


def effective_N(family, L, N=None):
    """Number of nodes of the given family: N = L * L for wsg, N (default L * L) for wsr."""
    if N is None:
        return L * L
    if family == "wsg" and N != L * L:
        raise ValueError(f"family wsg has N = L * L = {L * L}; --N {N} is not possible")
    return N


def make_graphs(family, k, p, G, L, N, rng):
    """G network realisations of the given family, degree and rewiring probability."""
    if family == "wsg":
        lattice = torus_lattice(L, k)
        return [ws_rewire_grid(lattice, p, rng) for _ in range(G)]
    return [ws_ring(N, k, p, seed=int(rng.integers(2**31))) for _ in range(G)]


def sweep(family, k, seed, out, L, N, G, M, T, rule, log):
    """Both sweeps for one (family, degree), with checkpointing."""
    rng = np.random.default_rng(seed)
    f_d = out / f"{family}_k{k}_dens.npy"
    f_p = out / f"{family}_k{k}_prob.npy"
    dens_res = np.load(f_d) if f_d.exists() else np.full(len(DENS_ARRAY), np.nan)
    prob_res = np.load(f_p) if f_p.exists() else np.full(len(PROB_ARRAY), np.nan)
    # rho^0 sweep at p = P_DENS (one set of realisations for all rho^0)
    if np.isnan(dens_res).any():
        graphs = make_graphs(family, k, P_DENS, G, L, N, rng)
        for i, d in enumerate(DENS_ARRAY):
            if not np.isnan(dens_res[i]):
                continue
            t0 = time.time()
            dens_res[i] = success_rate(graphs, rule, d, M, T, rng)
            np.save(f_d, dens_res)
            log(f"{family} k={k} rho0={d:.2f}: {dens_res[i]:.3f} ({time.time() - t0:.0f}s)")
    # p sweep at rho^0 = DENS_PROB (new realisations for every p)
    for i, p in enumerate(PROB_ARRAY):
        if not np.isnan(prob_res[i]):
            continue
        t0 = time.time()
        graphs = make_graphs(family, k, p, G, L, N, rng)
        prob_res[i] = success_rate(graphs, rule, DENS_PROB, M, T, rng)
        np.save(f_p, prob_res)
        log(f"{family} k={k} p={p:.3f}: {prob_res[i]:.3f} ({time.time() - t0:.0f}s)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--family", choices=["wsg", "wsr"], required=True)
    parser.add_argument("--degrees", type=int, nargs="+", help="mean degrees (default: 8 12 4 for wsg, 8 10 6 for wsr)")
    parser.add_argument("--seed", type=int, help="seed base; the sweep for degree k is seeded with base + k (default 100 for wsg, 200 for wsr)")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "fssp" / "thesis-2026")
    parser.add_argument("--L", type=int, default=30, help="side of the L x L network (default 30)")
    parser.add_argument("--N", type=int, help="number of nodes (default L * L; only wsr can differ from L * L)")
    parser.add_argument("--graphs", type=int, default=30, help="network realisations per point (default 30)")
    parser.add_argument("--configs", type=int, default=30, help="initial configurations per realisation (default 30)")
    parser.add_argument("--T", type=int, help="maximum number of time steps (default 2 N)")
    parser.add_argument("--beta", type=int, default=23)
    parser.add_argument("--sigma", type=int, default=47)
    args = parser.parse_args(argv)

    degrees = args.degrees or DEFAULT_DEGREES[args.family]
    seed_base = args.seed if args.seed is not None else DEFAULT_SEED_BASE[args.family]
    try:
        N = effective_N(args.family, args.L, args.N)
    except ValueError as exc:
        parser.error(str(exc))
    T = args.T if args.T is not None else 2 * N
    rule = Rule(9, args.beta, args.sigma)
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "log.txt", "a", encoding="utf-8") as logfile:
        def log(msg):
            logfile.write(time.strftime("%H:%M:%S ") + msg + "\n")
            logfile.flush()
            print(msg, flush=True)
        log(f"start {args.family} degrees={degrees} seed_base={seed_base} L={args.L} N={N} G={args.graphs} M={args.configs} T={T} rule={rule}")
        for k in degrees:
            sweep(args.family, k, seed_base + k, args.out, args.L, N, args.graphs, args.configs, T, rule, log)
        log("done")


if __name__ == "__main__":
    main()
