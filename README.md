# 🚀 Essential Metrics for Life on Graphs

> This project contains all the scripts and data that were developed and used for the ["Physica D" publication](https://authors.elsevier.com/sd/article/S0167-2789(25)00427-0) of the manuscript with the same name.

---

## 📖 Table of Contents
- [About](#-about)
    - [Abstract of the Publication](#abstract-of-the-publication)
    - [Nature of this Repo](#nature-of-this-repo)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## 🧐 About

#### Abstract of the Publication

We present a strong theoretical foundation that frames a well-defined family of outer-totalistic network automaton models as a topological generalisation of binary outer-totalistic cellular automata, of which the Game of Life is one notable particular case. These "Life-like network automata" are quantitatively described by expressing their genotype (the mean field curve and Derrida curve) and phenotype (the evolution of the state and defect averages). After demonstrating that the genotype and phenotype are correlated, we illustrate the utility of these essential metrics by tackling the firing squad synchronisation problem in a bottom-up fashion, with results that exceed a 90% success rate.

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

## 📂 Project structure
```
essential-metrics-life-on-graphs/
├─ src/
    └─ essential_metrics_life_on_graphs/
        ├─ analysis.py
        ├─ automata.py
        ├─ datasets.py
        ├─ networks.py
        ├─ rules.py
        ├─ simulation.py
        └─ visual.py
├─ data/
├─ notebooks/
    └─ essential-metrics_figures.ipynb
├─ README.md
├─ environment.yml
├─ .gitignore
└─ pyproject.toml
```

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