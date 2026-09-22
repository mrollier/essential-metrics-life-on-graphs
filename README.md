# 🚀 Essential Metrics for Life on Graphs

> This project contains all the scripts and data that were developed and used for the ["Physica D" publication](https://authors.elsevier.com/sd/article/S0167-2789(25)00427-0) of the manuscript with the same name.

---

## 📖 Table of Contents
- [About](#-about)
    - [Abstract of the Publication](#abstract-of-the-publication)
    - [Nature of this Repo](#nature-of-this-repo)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [Corrections (September 2026)](#-corrections-september-2026)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## 🧐 About

#### Abstract of the Publication

We present a strong theoretical foundation that frames a well-defined family of outer-totalistic network automaton models as a topological generalisation of binary outer-totalistic cellular automata, of which the Game of Life is one notable particular case. These "Life-like network automata" are quantitatively described by expressing their genotype (the mean field curve and Derrida curve) and phenotype (the evolution of the state and defect averages). After demonstrating that the genotype and phenotype are correlated, we illustrate the utility of these essential metrics by tackling the firing squad synchronisation problem in a bottom-up fashion, with results that exceed a 90% success rate.

> **Note on the last sentence.** The published figure shows a success rate of 84 % (ring lattices, N = 900, mean degree 8, rewiring probability p = 0.2). The corrected experiment on the grid-based networks that the text describes gives 96–98 % at p = 0.2; see [Corrections (September 2026)](#-corrections-september-2026).

#### Nature of this Repo

This is a minimal repository that contains all figures used in the publication and most of the data generation that was required for making these figures. Two files are currently in `.gitignore` due to their size: `data/final_defect_densities.h5` and `data/final_state_densities.h5`. You may contact the author (michiel.rollier@ugent.be) if you would like to gain access to these data, or you may of course generate them yourselves by enabling the required Booleans (`RUN_AGAIN=True`). Note that creating these data can take up to several hours when running on a commercial PC.

---

## 🚦 Getting Started

Clone the repo and install all dependencies with the following commands:

```bash
git clone https://github.com/mrollier/essential-metrics-life-on-graphs.git
cd your-repo
conda env create -f environment.yml
conda activate essential-metrics
```

Next, run through the Jupyter Notebook in `/notebooks`, enabling the relevant Booleans (`LOADDATA=True`, all others false).

The test suite (which also checks the numpy implementation of the LLNA against the torch implementation) runs with

```bash
python -m pytest
```

The scripts in `/scripts` reproduce the corrected synchronisation-task experiment; see `data/fssp/thesis-2026/README.md`.

## 📂 Project structure
```
essential-metrics-life-on-graphs/
├─ src/
    └─ essential_metrics_life_on_graphs/
        ├─ analysis.py
        ├─ automata.py
        ├─ datasets.py
        ├─ llna_numpy.py
        ├─ networks.py
        ├─ rules.py
        ├─ simulation.py
        └─ visual.py
├─ scripts/
    ├─ fssp_candidates.py
    ├─ fssp_sweep.py
    ├─ fssp_screening.py
    └─ fssp_plot.py
├─ tests/
├─ data/
    └─ fssp/
        ├─ paper-2025/      (published data, archived)
        └─ thesis-2026/     (corrected data)
├─ figures/
    └─ paper-2025/          (published synchronisation figure, archived)
├─ notebooks/
    └─ essential-metrics_figures.ipynb
├─ README.md
├─ environment.yml
├─ .gitignore
└─ pyproject.toml
```

## 🔧 Corrections (September 2026)

The synchronisation-task figure of the publication, `fssp-succes_rate-degrees7_8_9-N900-T1800.pdf`
(now in `figures/paper-2025/`), is affected by an error in the network construction.

- The three curves labelled ⟨k⟩ = 7, 8, 9 were computed on one-dimensional ring lattices
  generated with `igraph.Graph.Watts_Strogatz(dim=1, size=900, nei=k//2, p=p)`, of actual degree
  6, 8 and 8 respectively, and not on the grid-based (toroidal, Moore-neighbourhood) networks
  that the text describes. The curves labelled 8 and 9 are two samples of the same distribution.
- The published data are archived unchanged in `data/fssp/paper-2025/`, with a README that
  documents their provenance. The generating script is not available.
- The published success rate of 84 % (85 % on re-run) is the ring-lattice number for N = 900,
  ⟨k⟩ = 8, p = 0.2.
- On the grid-based networks the same rule R9B23S47 synchronises 96–98 % of the samples at
  p = 0.2 and at least 95 % for 0.02 ≤ p ≤ 0.4 (⟨k⟩ = 8); with ⟨k⟩ = 4 it never synchronises
  and with ⟨k⟩ = 12 it always does. The corrected data, both for the grids (⟨k⟩ = 4, 8, 12) and
  for the ring lattices (⟨k⟩ = 6, 8, 10), are in `data/fssp/thesis-2026/`; the corrected figures
  are `figures/fssp-success_rate-wsg-degrees4_8_12-N900-T1800.pdf` and
  `figures/fssp-success_rate-wsr-degrees6_8_10-N900-T1800.pdf`.
- The notebook cell that plotted the figure applied a quick fix to the entries at initial
  density 0 and 1; with a success criterion that includes t = 0 those entries are 1 by
  construction, and the fix has been removed.

The state of the repository as used for the publication is tagged `paper-2025`; the corrected
experiment as used in the PhD thesis of the first author (Ghent University, 2026) is tagged
`thesis-2026`. The publication itself is not changed.

## 📜 License
This project is licensed under the MIT license.

## 🙌 Acknowledgements

This work has been partially supported by
- the FWO grant with project title “An analysis of network automata as models for biological and natural processes” [3G0G0122];
- the FWO travel grant with file name V412625N;
- the FWO congress participation grant K105625N;
- the FAPESP grant #2024/02727-0;
- the CAPES grant #88887.841805/2023-00.

The authors wish to thank Gisele H. B. Miranda and Bernard De Baets for their invaluable contribution to the mathematical foundations of LLNAs.