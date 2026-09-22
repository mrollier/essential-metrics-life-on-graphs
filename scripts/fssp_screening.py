"""
Screen all candidate rules for the synchronisation task on rewired Moore grids
(wsg, k = 8, p = 0.2, rho^0 = 1/2): success rate over G realisations x M initial
configurations, at most T = 2 N time steps.

Reads <out>/candidates_r9_k8.json (see fssp_candidates.py) and writes
<out>/screening_wsg_k8_p0.2.json, checkpointing after every rule. The one set of network
realisations is generated from the seed (canonical: 1000) and shared by all rules.

Usage:
    python scripts/fssp_screening.py [--seed 1000] [--out data/fssp/thesis-2026]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "essential_metrics_life_on_graphs"))

from llna_numpy import Rule, success_rate  # noqa: E402
from networks import torus_lattice, ws_rewire_grid  # noqa: E402

K, P, DENS = 8, 0.2, 0.5


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "fssp" / "thesis-2026")
    parser.add_argument("--L", type=int, default=30)
    parser.add_argument("--graphs", type=int, default=30)
    parser.add_argument("--configs", type=int, default=30)
    parser.add_argument("--T", type=int, help="maximum number of time steps (default 2 N)")
    args = parser.parse_args(argv)
    T = args.T if args.T is not None else 2 * args.L * args.L

    f_out = args.out / f"screening_wsg_k{K}_p{P}.json"
    done = json.load(open(f_out, encoding="utf-8")) if f_out.exists() else {}
    cands = json.load(open(args.out / "candidates_r9_k8.json", encoding="utf-8"))
    rng = np.random.default_rng(args.seed)
    lattice = torus_lattice(args.L, K)
    graphs = [ws_rewire_grid(lattice, P, rng) for _ in range(args.graphs)]
    for beta, sigma in cands:
        key = f"{beta},{sigma}"
        if key in done:
            continue
        t0 = time.time()
        done[key] = success_rate(graphs, Rule(9, beta, sigma), DENS, args.configs, T, rng)
        with open(f_out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(done, f, indent=1)
        print(f"screening {key}: {done[key]:.3f} ({time.time() - t0:.0f}s)", flush=True)
    print(f"{len(done)} rules in {f_out}")


if __name__ == "__main__":
    main()
