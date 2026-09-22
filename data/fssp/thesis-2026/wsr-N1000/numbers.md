# Headline numbers: ring lattices at N = 1000

Rule R9B23S47 on the global synchronisation task, on `igraph.Graph.Watts_Strogatz` ring
lattices with N = 1000 nodes and T = 2N = 2000 time steps, 30 network realisations x 30
initial configurations per point. These are the numbers of the thesis figure
`figures/fssp-success_rate-wsr-degrees6_8_10-N1000-T2000.pdf`, whose left panel is the
rho^0 sweep at p = 0.2 and whose right panel is the p sweep at rho^0 = 1/2.

Percentages are rounded to the nearest integer; p values are the grid values (see the notes).

| # | quantity | value |
|---|---|---|
| a | screening: best rule and rate | (beta, sigma) = (23, 47), i.e. R9B23S47 — 83 % |
| a | screening: runner-up and rate | (79, 27) — 65 % |
| a | screening: rules at exactly 0 | 22 of the 27 candidates |
| a | screening: rules above 10 % | 3 — (23, 47) 83 %, (79, 27) 65 %, (47, 23) 16 % |
| b | k = 8, rho^0 sweep: rate at rho^0 = 0.5 | 78 % |
| b | k = 8, rho^0 sweep: rho^0 with rate >= 90 % | 0 <= rho^0 <= 0.20 and 0.80 <= rho^0 <= 1 |
| c | k = 6, rho^0 sweep: rate at rho^0 = 0.05 | 43 % |
| c | k = 6, rho^0 sweep: rate at rho^0 = 0.95 | 42 % |
| c | k = 6, rho^0 sweep: rho^0 with rate 0 | 0.25 <= rho^0 <= 0.75 |
| d | k = 10, rho^0 sweep: minimum rate | 99 % (0.994, at rho^0 = 0.70) |
| e | k = 8, p sweep: largest p with rate 0 | p = 0.0398 |
| e | k = 8, p sweep: rate at p = 0.2 | 84 % |
| e | k = 8, p sweep: maximum rate and its p | 94 % at p = 0.3162 |
| e | k = 8, p sweep: rate at p = 1 | 39 % |
| f | k = 10, p sweep: smallest p with rate >= 99 % | p = 0.1585 (rate 99.7 %) |
| f | k = 10, p sweep: rate at p = 1 | 97 % |
| g | k = 6, p sweep: maximum rate | 0 % (zero at every p of the grid) |
| h | (k = 8, p = 0.2, rho^0 = 1/2): screening | 83 % |
| h | (k = 8, p = 0.2, rho^0 = 1/2): left panel at rho^0 = 0.5 | 78 % |
| h | (k = 8, p = 0.2, rho^0 = 1/2): right panel at p = 0.2 | 84 % |
| h | (k = 8, p = 0.2, rho^0 = 1/2): spread | 6 percentage points |

## Notes

- The p sweep uses the grid `numpy.logspace(-2, 0, 21)`, whose point nearest 0.2 is
  p = 0.19953; the entries above labelled p = 0.2 in the p sweep are that grid point. The
  rho^0 sweep and the screening are at exactly p = 0.2.
- The range in (b) is a union of two intervals, not one: the rate dips below 90 % in the
  middle of the rho^0 range and reaches its minimum of 78 % at rho^0 = 0.5.
- The rate in (f) is not monotone in p: it first reaches 99 % at p = 0.1585 and stays there
  up to p = 0.5012, then falls back to 97 %, 93 % and 97 % at the last three grid points.
- The three estimates in (h) are three independent samples of the same quantity, each on its
  own set of 30 network realisations, so the 6 percentage point spread is sampling scatter.
