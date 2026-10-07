"""Synthetic high-resolution manometry (HRM) swallows with known ground truth.

A swallow is a pressure array P[t, z] (mmHg) sampled at 20 Hz on 36 sensors 1 cm apart. Pressures are
referenced to the stomach (gastric pressure ~ 0). Layout (cm from the top sensor):
    0-2    upper oesophageal sphincter (UES)
    3-8    proximal (striated-muscle) body;  ~9-10  transition zone (pressure trough)
    11-29  distal (smooth-muscle) body
    30-34  oesophago-gastric junction (EGJ / LES), centre 32;  35  stomach
The UES relaxes at t_ues = 2 s. Each sensor contracts as a smooth pulse whose onset travels down the
oesophagus quickly to the contractile deceleration point (CDP) and slowly after it, so the distal latency
DL = t_onset(CDP) - t_ues is a parameter. The EGJ relaxes after the swallow to a set nadir, which sets IRP.

This is a phenomenological generator for testing analysis software, not a physiological model (for that,
see Miura et al., R Soc Open Sci 2025).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FS = 20.0                  # Hz
N_SENSORS = 36
DURATION = 25.0            # s
T_UES = 2.0                # s
Z_TZ = 9.5                 # transition zone (cm)
Z_CDP = 27.0               # contractile deceleration point (cm)
Z_EGJ = 32.0               # EGJ centre (cm)


@dataclass(frozen=True)
class SwallowSpec:
    """Ground-truth parameters of one synthetic swallow."""

    distal_amp: float = 120.0     # peak distal contraction (mmHg)
    duration: float = 3.5         # contraction duration at each sensor (s)
    dl: float = 6.5               # distal latency (s)
    egj_nadir: float = 3.0        # EGJ pressure during deglutitive relaxation (mmHg)
    gap: tuple | None = None      # (z_start, z_end) of a contraction gap in the body (cm)
    pan: float = 0.0              # panesophageal pressurisation level (mmHg), 0 = none
    ibp: float = 0.0              # intrabolus pressure ahead of the wave (mmHg)
    noise: float = 1.5            # sensor noise (mmHg, SD)


def swallow(spec: SwallowSpec, rng: np.random.Generator | None = None) -> np.ndarray:
    """Return P[t, z] for one swallow (shape: 500 x 36)."""
    rng = rng or np.random.default_rng()
    t = np.arange(0, DURATION, 1 / FS)[:, None]
    z = np.arange(N_SENSORS, dtype=float)[None, :]
    resp = 2.0 * np.sin(2 * np.pi * t / 4.0)
    p = np.zeros((t.shape[0], N_SENSORS)) + resp

    # oesophageal body baseline (slightly above gastric)
    body = (z >= 3) & (z <= 29)
    p = p + 2.0 * body

    # UES: tonic high pressure, relaxes at T_UES, then a post-swallow contraction
    ues = (z <= 2)
    ues_relax = _window(t, T_UES, T_UES + 0.6, 0.1)
    p = p + ues * (70 * (1 - ues_relax) + 50 * _pulse(t, T_UES + 0.9, 0.8))

    # peristaltic onset times: fast to the CDP, slow after it. A sin^2 contraction crosses the 30 mmHg
    # contour ~0.55 s after its onset for typical amplitudes, so onsets are shifted to put that crossing at DL.
    lag = 0.55
    v1 = (Z_CDP - 3.0) / max(spec.dl - lag - 0.5, 0.3)
    onset = np.where(z <= Z_CDP, T_UES + 0.5 + (z - 3.0) / v1, T_UES + spec.dl - lag + (z - Z_CDP) / 1.5)
    amp = np.where(z < Z_TZ - 1.5, 60.0, np.where(z <= Z_TZ + 1.0, 12.0, spec.distal_amp))
    taper = np.clip((z - Z_TZ) / 4.0, 0.6, 1.0)                    # amplitude builds up distally
    amp = np.where(z > Z_TZ + 1.0, amp * taper, amp)
    if spec.gap is not None:
        g0, g1 = spec.gap
        amp = np.where((z >= g0) & (z <= g1), 5.0, amp)
    contraction = amp * _sin2(t, onset, spec.duration)
    p = p + ((z >= 3) & (z <= 29)) * contraction

    # intrabolus pressure ahead of the wave and panesophageal pressurisation
    if spec.ibp > 0:
        ahead = (t > T_UES + 0.5) & (t < onset) & (z >= 11) & (z <= 29)
        p = p + spec.ibp * ahead
    if spec.pan > 0:
        p = p + ((z >= 3) & (z <= 34)) * spec.pan * _window(t, T_UES + 1.0, T_UES + 6.0, 0.4)

    # EGJ: tonic ~25 mmHg, relaxes to egj_nadir for ~8 s after the swallow, then an after-contraction
    egj_profile = np.exp(-0.5 * ((z - Z_EGJ) / 1.3) ** 2) * (z >= 29)
    relax = _window(t, T_UES + 0.3, T_UES + spec.dl + 3.0, 0.4)
    egj = 25.0 * (1 - relax) + spec.egj_nadir * relax
    after = 0.6 * spec.distal_amp * _pulse(t, T_UES + spec.dl + 4.0, 4.0) * (spec.distal_amp > 30)
    p = p + egj_profile * (egj + after)

    return p + spec.noise * rng.standard_normal(p.shape)


def _pulse(t, centre, width):
    """Smooth bump of height 1 and roughly `width` seconds duration."""
    return np.exp(-0.5 * ((t - centre) / (width / 2.6)) ** 2)


def _sin2(t, onset, duration):
    """Contraction starting exactly at `onset`: sin^2 bump lasting `duration` seconds."""
    x = (t - onset) / duration
    return np.where((x >= 0) & (x <= 1), np.sin(np.pi * np.clip(x, 0, 1)) ** 2, 0.0)


def _window(t, start, stop, ramp):
    """1 between start and stop with smooth logistic edges of time constant `ramp`."""
    return 1 / (1 + np.exp(-(t - start) / ramp)) * 1 / (1 + np.exp((t - stop) / ramp))


# ---- swallow types and study-level phenotypes ---------------------------------------------------
TYPES = {
    "normal": dict(),
    "weak": dict(distal_amp=27.0),
    "failed": dict(distal_amp=8.0),
    "fragmented": dict(gap=(14, 21)),
    "premature": dict(dl=3.2),
    "hypercontractile": dict(distal_amp=320.0, duration=6.0),
    "pan": dict(distal_amp=8.0, pan=45.0),
    "spastic": dict(dl=3.0, distal_amp=150.0),
}

# (swallow types for 10 supine swallows, EGJ nadir in mmHg, intrabolus pressure)
PHENOTYPES = {
    "normal": (["normal"] * 9 + ["weak"], 3.0, 0.0),
    "achalasia I": (["failed"] * 10, 24.0, 0.0),
    "achalasia II": (["pan"] * 6 + ["failed"] * 4, 26.0, 0.0),
    "achalasia III": (["spastic"] * 7 + ["failed"] * 3, 25.0, 0.0),
    "EGJ outflow obstruction": (["normal"] * 10, 21.0, 20.0),
    "absent contractility": (["failed"] * 10, 4.0, 0.0),
    "distal oesophageal spasm": (["premature"] * 4 + ["normal"] * 6, 3.0, 0.0),
    "hypercontractile oesophagus": (["hypercontractile"] * 4 + ["normal"] * 6, 3.0, 0.0),
    "ineffective oesophageal motility": (["weak"] * 5 + ["failed"] * 2 + ["fragmented"] * 1 + ["normal"] * 2, 3.0, 0.0),
}


def study(phenotype: str, seed: int = 0, jitter: float = 0.15):
    """Ten supine swallows of a phenotype, with per-swallow random variation.

    Returns (swallows, truth) where swallows is a list of P[t, z] arrays and truth the swallow types.
    """
    rng = np.random.default_rng(seed)
    types, nadir, ibp = PHENOTYPES[phenotype]
    types = list(rng.permutation(types))
    out = []
    for typ in types:
        kw = dict(TYPES[typ])
        kw["distal_amp"] = kw.get("distal_amp", 120.0) * (1 + jitter * rng.uniform(-1, 1))
        kw["dl"] = kw.get("dl", 6.5) + 0.4 * rng.uniform(-1, 1)
        kw["egj_nadir"] = nadir + 1.5 * rng.uniform(-1, 1)
        kw["ibp"] = ibp
        out.append(swallow(SwallowSpec(**kw), rng))
    return out, types
