"""
Screen all candidate rules for the synchronisation task on one network family at k = 8,
p = 0.2, rho^0 = 1/2: success rate over G realisations x M initial configurations, at most
T = 2 N time steps.

Two families, built exactly as in `scripts/fssp_sweep.py`:
  wsg  rewired toroidal Moore grids, N = L * L (the default, N = 900);
  wsr  igraph Watts_Strogatz ring lattices of N nodes (the thesis uses N = 1000).

Reads <out>/candidates_r9_k8.json (see fssp_candidates.py) and writes
<out>/screening_<family>_k8_p0.2.json, checkpointing after every rule. The one set of network
realisations is generated from the seed (canonical: 1000) and shared by all rules.

Usage:
    python scripts/fssp_screening.py [--seed 1000] [--out data/fssp/thesis-2026]
    python scripts/fssp_screening.py --family wsr --N 1000 --seed 1300 \
        --out data/fssp/thesis-2026/wsr-N1000
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "essential_metrics_life_on_graphs"))
sys.path.insert(0, str(ROOT / "scripts"))

from llna_numpy import Rule, success_rate  # noqa: E402
from fssp_sweep import effective_N, make_graphs  # noqa: E402

K, P, DENS = 8, 0.2, 0.5


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--family", choices=["wsg", "wsr"], default="wsg")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "fssp" / "thesis-2026")
    parser.add_argument("--L", type=int, default=30, help="side of the L x L network (default 30)")
    parser.add_argument("--N", type=int, help="number of nodes (default L * L; only wsr can differ from L * L)")
    parser.add_argument("--graphs", type=int, default=30)
    parser.add_argument("--configs", type=int, default=30)
    parser.add_argument("--T", type=int, help="maximum number of time steps (default 2 N)")
    args = parser.parse_args(argv)
    try:
        N = effective_N(args.family, args.L, args.N)
    except ValueError as exc:
        parser.error(str(exc))
    T = args.T if args.T is not None else 2 * N

    args.out.mkdir(parents=True, exist_ok=True)
    f_out = args.out / f"screening_{args.family}_k{K}_p{P}.json"
    done = json.load(open(f_out, encoding="utf-8")) if f_out.exists() else {}
    f_cands = args.out / "candidates_r9_k8.json"
    if not f_cands.exists():  # a fresh output directory shares the canonical candidate list
        f_cands = ROOT / "data" / "fssp" / "thesis-2026" / "candidates_r9_k8.json"
    cands = json.load(open(f_cands, encoding="utf-8"))
    rng = np.random.default_rng(args.seed)
    graphs = make_graphs(args.family, K, P, args.graphs, args.L, N, rng)
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
