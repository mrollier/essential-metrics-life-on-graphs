"""
Recompute the candidate rules for the synchronisation task and write them to
data/fssp/thesis-2026/candidates_r9_k8.json.

A rule of resolution 9 is a candidate if
  * it is self-equivalent (equal to its own equivalent rule, see rules.return_equivalent_rule),
  * its mean-field curve f satisfies f(rho) >= 1 - rho for every rho on a density grid in
    (0, 1/2) at degree 8 (self-equivalence then gives f(rho) <= 1 - rho for rho > 1/2), and
  * it is not the pure-complementation rule (511, 0), whose mean-field curve is identically
    1 - rho: the non-strict inequality admits it, but it cannot synchronise anything.

Usage:
    python scripts/fssp_candidates.py [--degree 8] [--points 101] [--tol 1e-9] [--out PATH]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "essential_metrics_life_on_graphs"))

from llna_numpy import Rule  # noqa: E402

RESOLUTION = 9
ALL = (1 << RESOLUTION) - 1


def reverse_bits(n):
    """Mirror the RESOLUTION-bit masks in n (interval k -> RESOLUTION - 1 - k)."""
    out = np.zeros_like(n)
    for k in range(RESOLUTION):
        out |= ((n >> k) & 1) << (RESOLUTION - 1 - k)
    return out


def self_equivalent_rules():
    """All (beta, sigma) pairs of resolution 9 that equal their own equivalent rule."""
    beta = np.arange(1 << RESOLUTION)[:, None]
    sigma = np.arange(1 << RESOLUTION)[None, :]
    mask = (beta == (~reverse_bits(sigma) & ALL)) & (sigma == (~reverse_bits(beta) & ALL))
    b, s = np.nonzero(mask)
    return sorted(zip(b.tolist(), s.tolist()))


def mean_field_curve(rule, degree, rho):
    """Mean-field density after one step, for nodes of the given degree (isotropic rule)."""
    q = np.arange(degree + 1)
    k = rule.lut[degree, q]
    pmf = binom.pmf(q, degree, rho[:, None])
    born = rule.B[k] * (1 - rho)[:, None]
    survive = rule.S[k] * rho[:, None]
    return np.sum(pmf * (born + survive), axis=1)


def candidate_rules(degree=8, num_points=101, tol=1e-9):
    rho = np.linspace(0, 1, num_points)[1:-1]
    lower = rho < 0.5
    out = []
    for beta, sigma in self_equivalent_rules():
        if (beta, sigma) == (ALL, 0):
            continue
        f = mean_field_curve(Rule(RESOLUTION, beta, sigma, max_deg=degree), degree, rho)
        if np.all(f[lower] >= 1 - rho[lower] - tol):
            out.append([beta, sigma])
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--degree", type=int, default=8)
    parser.add_argument("--points", type=int, default=101, help="number of points of the density grid on [0, 1]")
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "fssp" / "thesis-2026" / "candidates_r9_k8.json")
    args = parser.parse_args(argv)
    cands = candidate_rules(args.degree, args.points, args.tol)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(cands, f)
    print(f"{len(cands)} candidate rules written to {args.out}")


if __name__ == "__main__":
    main()
