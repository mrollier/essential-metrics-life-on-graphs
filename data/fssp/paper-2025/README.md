# Published synchronisation-task data (Rollier et al., 2025)

These two arrays are the data behind the synchronisation figure of Rollier et al. (2025),
*Essential metrics for Life on graphs*, Physica D
(`figures/paper-2025/fssp-succes_rate-degrees7_8_9-N900-T1800.pdf`). They are kept here,
unchanged, so that the published figure remains reproducible. The corrected experiment used in
the PhD thesis is in `data/fssp/thesis-2026/`; see the section "Corrections (September 2026)"
in the top-level `README.md`.

## Files

| file | shape | sweep |
|---|---|---|
| `success_rates_init_dens.npy` | 3 x 21 | success rate against initial density rho^0, at p = 0.2 |
| `success_rates_rewiring_probs.npy` | 3 x 21 | success rate against rewiring probability p, at rho^0 = 1/2 |

Rule: R9B23S47 (phi^9_{23,47}), the isotropic (`iso=True`) local update rule of resolution 9.

## What the rows are

The rows were labelled <k> = 7, 8 and 9 in the notebook and in the paper. The script that
generated them is not available. Reconstruction shows that the networks were produced with

    igraph.Graph.Watts_Strogatz(dim=1, size=900, nei=k//2, p=p)

for k = 7, 8, 9, i.e. one-dimensional ring lattices, with `nei = 7//2 = 3`, `8//2 = 4` and
`9//2 = 4`. The actual mean degrees of the three rows are therefore

| row | label in the figure | actual degree |
|---|---|---|
| 0 | <k> = 7 | 6 |
| 1 | <k> = 8 | 8 |
| 2 | <k> = 9 | 8 |

Rows 1 and 2 are two samples of the same distribution, which is why the two curves coincide.
None of the rows is the grid-based (toroidal, Moore-neighbourhood) family that the text of the
paper and `networks.create_2d_torus_lattice` suggest. A numpy re-implementation of the LLNA
(`src/essential_metrics_life_on_graphs/llna_numpy.py`) reproduces these numbers on ring lattices
(e.g. 85.0 % against the stored 85.6 % at p = 0.2, rho^0 = 1/2; 0 % everywhere for degree 6).

## Parameters

- N = 900 nodes, T = 1800 time steps (T = 2N).
- rho^0 grid: `numpy.linspace(0, 1, 21)`, at p = 0.2 (`success_rates_init_dens.npy`).
- p grid: `numpy.logspace(-2, 0, 21)`, at rho^0 = 1/2 (`success_rates_rewiring_probs.npy`).
- 900 samples per point (inferred: every stored value is an exact multiple of 1/900), presumably
  30 network realisations x 30 initial configurations.
- Success: the configuration is homogeneous (all 0 or all 1) at some t <= T.

## Quick fix applied at plotting time

The notebook cell that plotted the figure contained a `QUICKFIX` that set the entries at
rho^0 = 0 and rho^0 = 1 of `success_rates_init_dens.npy` to 1 before plotting. The stored
values at those two densities are 0.9667 and 0.9 (row 0) and 1 (rows 1 and 2). With a success
criterion that includes t = 0, the homogeneous initial configurations are counted as successes
and those entries are 1 automatically; the stored values indicate that the original script did
not count t = 0. The arrays here are stored as they were, without the quick fix.
