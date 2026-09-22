# Corrected synchronisation-task data (PhD thesis, 2026)

Success rate of rule R9B23S47 (phi^9_{23,47}) on the global synchronisation task, re-run with
the numpy implementation of the LLNA in `src/essential_metrics_life_on_graphs/llna_numpy.py`.
These are the canonical data of the thesis figure and of the figures
`figures/fssp-success_rate-wsg-degrees4_8_12-N900-T1800.pdf` and
`figures/fssp-success_rate-wsr-degrees6_8_10-N900-T1800.pdf`. The published data of
Rollier et al. (2025) are archived in `data/fssp/paper-2025/`.

## Files

| file | content |
|---|---|
| `wsg_k{4,8,12}_dens.npy` | success rate against rho^0 (21 values) on rewired Moore-type grids, p = 0.2 |
| `wsg_k{4,8,12}_prob.npy` | success rate against p (21 values) on rewired grids, rho^0 = 1/2 |
| `wsr_k{6,8,10}_dens.npy` | as above, on igraph ring lattices |
| `wsr_k{6,8,10}_prob.npy` | as above, on igraph ring lattices |
| `candidates_r9_k8.json` | the 27 resolution-9 candidate rules (`scripts/fssp_candidates.py`) |
| `screening_wsg_k8_p0.2.json` | success rate of every candidate on wsg, k = 8, p = 0.2 (`scripts/fssp_screening.py`) |
| `log.txt` | timestamps and per-point results of the run |

## Network families

- **wsg**: toroidal 30 x 30 grid with a von Neumann (k = 4), Moore (k = 8) or radius-2 von
  Neumann (k = 12) neighbourhood (`networks.torus_lattice`), rewired with
  `networks.ws_rewire_grid`, which follows the repository's `watts_strogatz_rewire`: every edge
  is moved with probability p to a uniformly random node, disconnected results are rejected.
- **wsr**: `igraph.Graph.Watts_Strogatz(dim=1, size=900, nei=k//2, p=p)` ring lattices
  (`networks.ws_ring`), disconnected results rejected. This is the family behind the
  published figure.

## Parameters

- N = 900 nodes (L = 30), T = 2N = 1800 time steps.
- 30 network realisations x 30 initial configurations = 900 samples per point.
- rho^0 sweep: 21 equispaced values `numpy.linspace(0, 1, 21)` at p = 0.2, one set of 30
  realisations for all rho^0.
- p sweep: 21 log-spaced values `numpy.logspace(-2, 0, 21)` at rho^0 = 1/2, 30 new
  realisations per p.
- Every initial configuration has exactly round(rho^0 N) living nodes.
- Success: the configuration is homogeneous (all 0 or all 1) at some t <= T, t = 0 included.
  A configuration that revisits an earlier state has entered a cycle and is counted as a failure
  without running to T.
- Screening: all 27 candidates on the same 30 realisations of wsg, k = 8, p = 0.2, rho^0 = 1/2.

## Seeds and reproducibility

The numpy generator of each (family, degree) sweep was seeded with

| family | k = 4 | k = 6 | k = 8 | k = 10 | k = 12 |
|---|---|---|---|---|---|
| wsg | 104 | | 108 | | 112 |
| wsr | | 206 | 208 | 210 | |

i.e. seed base 100 (wsg) or 200 (wsr) plus k, which is what `scripts/fssp_sweep.py` does by
default. The screening used seed 1000. The wsg data are bitwise reproducible with that script.

Caveat for the ring family: igraph's generator (Python's `random` module) was **not** seeded
in the original run, so the wsr data are statistically, not bitwise, reproducible. The script
now draws a seed for every ring realisation from the numpy generator.

`scripts/fssp_sweep.py` checkpoints after every point; running it in this directory with all
checkpoints present computes nothing but still appends a start/done line to `log.txt`.

## Headline numbers

- wsg, k = 8: 97.8 % at p = 0.2; >= 95 % for 0.02 <= p <= 0.4; maximum 99.6 % at p = 0.04;
  82 % at p = 0.01; 44 % at p = 1. In the rho^0 sweep (p = 0.2): 96.3 % at rho^0 = 1/2, at
  least 96.3 % everywhere.
- wsg, k = 4: 0 everywhere (except the homogeneous initial configurations rho^0 = 0, 1).
- wsg, k = 12: 100 % everywhere (>= 98.7 % in the p sweep).
- wsr, k = 8: 80.9 % at p = 0.2 in the p sweep (85.0 % at rho^0 = 1/2 in the rho^0 sweep,
  which uses a different set of realisations); 0 for p < 0.05; maximum 91.9 % at p = 0.25;
  55.6 % at p = 1. This is the number behind the published 84 %.
- wsr, k = 6: 0 everywhere except near rho^0 = 0 and 1.
- wsr, k = 10: about 100 % for p >= 0.13; 0 for p < 0.05.
- Screening on wsg, k = 8, p = 0.2: (23, 47) 0.961; (79, 27) 0.613; (143, 29) 0.158;
  (47, 23) 0.140; (15, 31) 0.047; the remaining 22 rules at most 0.003, 19 of them exactly 0.
