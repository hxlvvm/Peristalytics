"""Tests of the CCv4 metric engine against synthetic swallows."""
import numpy as np
import pytest

from peristalytics import PHENOTYPES, SwallowSpec, analyse_swallow, diagnose_study, study, swallow
from peristalytics.metrics import FS, dci


def test_dci_of_a_constant_block_is_exact():
    p = np.zeros((int(25 * FS), 36))
    p[:, :3] = 70                                   # resting UES, relaxing at 2 s
    p[int(2 * FS):, :3] = 0
    p[int(4 * FS): int(6 * FS), 12:22] = 70         # 2 s x 10 cm at 70 mmHg -> (70 - 20) * 2 * 10 = 1000
    assert dci(p, t_ues=2.0, z_tz=10, z_egj=32) == pytest.approx(1000.0, rel=1e-6)


def test_irp_follows_egj_relaxation():
    rng = np.random.default_rng(0)
    for nadir in (3.0, 12.0, 25.0):
        m = analyse_swallow(swallow(SwallowSpec(egj_nadir=nadir, noise=0.5), rng))
        assert abs(m.irp - nadir) < 3.5


def test_distal_latency_is_recovered():
    rng = np.random.default_rng(1)
    for dl in (3.0, 5.0, 7.0):
        m = analyse_swallow(swallow(SwallowSpec(dl=dl), rng))
        assert abs(m.dl - dl) < 0.6


def test_landmarks_are_detected_from_the_data():
    m = analyse_swallow(swallow(SwallowSpec(), np.random.default_rng(2)))
    assert abs(m.t_ues - 2.0) < 0.2 and m.z_egj in (31, 32, 33) and 8 <= m.z_tz <= 12


@pytest.mark.parametrize("typ, expected", [("normal", "normal"), ("weak", "weak"), ("failed", "failed"),
                                           ("fragmented", "fragmented"), ("premature", "premature"),
                                           ("hypercontractile", "hypercontractile"), ("pan", "failed")])
def test_swallow_types(typ, expected):
    from peristalytics import TYPES
    m = analyse_swallow(swallow(SwallowSpec(**TYPES[typ]), np.random.default_rng(3)))
    assert m.swallow_type == expected


@pytest.mark.parametrize("phenotype", list(PHENOTYPES))
def test_every_phenotype_is_diagnosed(phenotype):
    hits = sum(diagnose_study(study(phenotype, seed=s)[0])["diagnosis"] == phenotype for s in range(3))
    assert hits == 3
