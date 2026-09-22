"""
Smoke test of the sweep driver on a tiny setting, and consistency of the canonical
candidate list on disk with the filter.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
THESIS_DATA = ROOT / "data" / "fssp" / "thesis-2026"


def test_sweep_driver_smoke(tmp_path):
    """L = 6, 2 realisations, 2 configurations, T = 20; the driver writes both arrays."""
    cmd = [sys.executable, str(SCRIPTS / "fssp_sweep.py"), "--family", "wsg", "--degrees", "8",
           "--L", "6", "--graphs", "2", "--configs", "2", "--T", "20", "--out", str(tmp_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr
    for name in ("wsg_k8_dens.npy", "wsg_k8_prob.npy"):
        arr = np.load(tmp_path / name)
        assert arr.shape == (21,)
        assert np.all((arr >= 0) & (arr <= 1)) and not np.isnan(arr).any()
    dens = np.load(tmp_path / "wsg_k8_dens.npy")
    assert dens[0] == 1 and dens[-1] == 1  # homogeneous initial configurations count as successes
    assert (tmp_path / "log.txt").read_text().strip().endswith("done")
    # a second run with complete checkpoints computes nothing and keeps the arrays
    before = np.load(tmp_path / "wsg_k8_prob.npy")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr
    assert np.array_equal(before, np.load(tmp_path / "wsg_k8_prob.npy"))


def test_candidates_on_disk_match_filter():
    """data/fssp/thesis-2026/candidates_r9_k8.json equals the recomputed candidate set."""
    from test_llna_numpy import CANDIDATES_R9_K8, candidate_rules
    on_disk = [tuple(c) for c in json.loads((THESIS_DATA / "candidates_r9_k8.json").read_text())]
    assert on_disk == sorted(CANDIDATES_R9_K8)
    assert set(on_disk) == candidate_rules()


def test_screening_keys_are_the_candidates():
    """The screening results cover exactly the candidate rules, with rates in [0, 1]."""
    cands = json.loads((THESIS_DATA / "candidates_r9_k8.json").read_text())
    screening = json.loads((THESIS_DATA / "screening_wsg_k8_p0.2.json").read_text())
    assert set(screening) == {f"{b},{s}" for b, s in cands}
    assert all(0 <= v <= 1 for v in screening.values())
    assert screening["23,47"] == max(screening.values())
