"""
Draw the two-panel synchronisation figure (success rate against rho^0 and against p) from
the data in a data directory, with the formatting of the notebook cell that produced the
published figure (LaTeX text rendering, figsize (10, 4), fontsize 22, viridis markers).

The file name follows the data: the number of nodes N and the number of time steps T are
read from the README.md of the data directory, or given with --N/--T. With the defaults,

    python scripts/fssp_plot.py

writes (from data/fssp/thesis-2026/, N = 900, T = 1800)
    figures/fssp-success_rate-wsg-degrees4_8_12-N900-T1800.pdf   (legend centre left)
    figures/fssp-success_rate-wsr-degrees6_8_10-N900-T1800.pdf   (legend upper left)
and the thesis figure is drawn from the ring data at N = 1000 with

    python scripts/fssp_plot.py --data data/fssp/thesis-2026/wsr-N1000 --families wsr \
        --N 1000 --T 2000 --png
        -> figures/fssp-success_rate-wsr-degrees6_8_10-N1000-T2000.pdf (and .png)

Note on fonts: the notebook sets font.serif to "Times New Roman" together with text.usetex.
Under usetex matplotlib does not recognise that name and LaTeX's default Computer Modern is
used; this is what the published figure shows and it is reproduced here.

Usage:
    python scripts/fssp_plot.py [--data DIR] [--figures DIR] [--families wsg wsr]
                                [--N N] [--T T] [--png] [--mathtext]
The --mathtext flag replaces LaTeX rendering by matplotlib's mathtext for machines without a
LaTeX installation; the output is then marked as such in the file name. The --png flag writes
a PNG next to the PDF.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colormaps, rcParams  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DENS_ARRAY = np.linspace(0, 1, 21)
PROB_ARRAY = np.logspace(-2, 0, 21)
FIGURES = {
    "wsg": ([4, 8, 12], "center left"),
    "wsr": ([6, 8, 10], "upper left"),
}


def read_size(data_dir):
    """
    (N, T) as stated in <data_dir>/README.md, each None if the README does not state it.

    The README of a data directory describes its run with a line such as
    "N = 900 nodes (L = 30), T = 2N = 1800 time steps."; this reads those two numbers so
    that the figure drawn from a directory is named after the data it shows.
    """
    readme = data_dir / "README.md"
    if not readme.exists():
        return None, None
    text = readme.read_text(encoding="utf-8")
    m_N = re.search(r"\bN\s*=\s*(\d+)\s+nodes", text)
    m_T = re.search(r"\bT\s*=\s*(?:2\s*N\s*=\s*)?(\d+)\s+time steps", text)
    return (int(m_N.group(1)) if m_N else None, int(m_T.group(1)) if m_T else None)


def draw(family, degrees, legend_loc, data_dir, fname):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    fontsize = 22
    markers = ["o", "^", "*"]
    markersize = 75
    cmap = colormaps["viridis"]
    colors = [cmap(0.), cmap(.4), cmap(.8)]

    for degree, marker, color in zip(degrees, markers, colors):
        label = f"$\\langle k\\rangle={degree}$"
        success_dens = np.load(data_dir / f"{family}_k{degree}_dens.npy")
        success_prob = np.load(data_dir / f"{family}_k{degree}_prob.npy")
        axs[0].scatter(DENS_ARRAY, success_dens, marker=marker, s=markersize, color=color, label=label)
        axs[1].scatter(PROB_ARRAY, success_prob, marker=marker, s=markersize, color=color, label=label)

    axs[0].set_ylim([-0.05, 1.05])
    yticks = [0, .25, .5, .75, 1]
    axs[0].set_yticks(yticks)
    axs[0].set_yticklabels(["0", "25", "50", "75", "100"], fontsize=fontsize - 2)
    axs[0].set_ylabel(r"Success rate (\%)" if rcParams["text.usetex"] else "Success rate (%)", fontsize=fontsize)
    xticks = [0, .2, .4, .6, .8, 1]
    axs[0].set_xticks(xticks)
    axs[0].set_xticklabels(xticks, fontsize=fontsize - 2)
    axs[0].set_xlim([-0.05, 1.05])
    axs[0].set_xlabel("Initial state average $\\rho^0$", fontsize=fontsize)

    axs[1].set_ylim([-0.05, 1.05])
    xticks = [0.01, 0.1, 1]
    axs[1].set_xticks(xticks)
    axs[1].set_xticklabels(["0.01", "0.1", "1"], fontsize=fontsize - 2)
    axs[1].set_xlim([0.009, 1.1])
    axs[1].set_xscale("log")
    axs[1].set_xlabel("Rewiring probability $p$", fontsize=fontsize)
    axs[1].legend(fontsize=fontsize - 6, loc=legend_loc)

    fig.tight_layout(w_pad=2)
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "fssp" / "thesis-2026")
    parser.add_argument("--figures", type=Path, default=ROOT / "figures")
    parser.add_argument("--N", type=int, help="number of nodes in the file name (default: from the README of --data)")
    parser.add_argument("--T", type=int, help="number of time steps in the file name (default: from the README of --data)")
    parser.add_argument("--png", action="store_true", help="also write a PNG next to the PDF")
    parser.add_argument("--mathtext", action="store_true", help="use mathtext instead of LaTeX")
    parser.add_argument("--families", nargs="+", choices=list(FIGURES), default=list(FIGURES))
    args = parser.parse_args(argv)

    readme_N, readme_T = read_size(args.data)
    N = args.N if args.N is not None else readme_N
    T = args.T if args.T is not None else readme_T
    if N is None or T is None:
        parser.error(f"{args.data / 'README.md'} does not state N and T; give them with --N and --T")

    if args.mathtext:
        rcParams.update({"text.usetex": False, "font.family": "serif", "mathtext.fontset": "cm"})
        suffix = "-mathtext"
    else:
        # identical to the notebook's set-up cell
        rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "text.latex.preamble": r"\usepackage{amsmath}",
        })
        suffix = ""
    args.figures.mkdir(parents=True, exist_ok=True)
    for family in args.families:
        degrees, legend_loc = FIGURES[family]
        stem = f"fssp-success_rate-{family}-degrees{'_'.join(map(str, degrees))}-N{N}-T{T}{suffix}"
        for ext in (".pdf", ".png") if args.png else (".pdf",):
            draw(family, degrees, legend_loc, args.data, args.figures / (stem + ext))
            print(f"written {args.figures / (stem + ext)}")


if __name__ == "__main__":
    main()
