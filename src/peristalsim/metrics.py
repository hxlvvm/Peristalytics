"""Chicago Classification v4.0 swallow metrics (IRP, DCI, DL, breaks) from P[t, z]."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FS = 20.0


@dataclass
class SwallowMetrics:
    irp: float
    dci: float
    dl: float           # NaN when there is no measurable contraction front
    break_cm: float
    pan: bool
    t_ues: float
    z_egj: int
    z_tz: int

    @property
    def swallow_type(self) -> str:
        """CCv4 swallow category."""
        if self.pan or self.dci < 100:
            return "failed"                      # pressurisation without a contraction counts as failed
        if self.dci < 450:
            return "weak"
        if self.dci > 8000:
            return "hypercontractile"
        if np.isfinite(self.dl) and self.dl < 4.5:
            return "premature"
        if self.break_cm > 5:
            return "fragmented"
        return "normal"


def ues_relaxation(p: np.ndarray, fs: float = FS) -> float:
    """Time (s) at which the UES pressure first drops below half its resting level."""
    ues = p[:, :3].max(axis=1)
    rest = np.median(ues[: int(fs)])
    below = np.flatnonzero(ues < 0.5 * rest)
    return float(below[0] / fs) if below.size else 0.0


def egj_location(p: np.ndarray, t_ues: float, fs: float = FS) -> int:
    """Sensor with the highest resting pressure in the distal third, before the swallow."""
    pre = p[: max(int(t_ues * fs), 1), :]
    lo = p.shape[1] * 2 // 3
    return int(lo + np.argmax(pre[:, lo:].mean(axis=0)))


def irp(p: np.ndarray, t_ues: float, z_egj: int, fs: float = FS, half_width: int = 2) -> float:
    """Integrated relaxation pressure (mmHg), e-sleeve over z_egj +/- half_width."""
    i0 = int(round(t_ues * fs))
    sleeve = p[i0: i0 + int(10 * fs), max(z_egj - half_width, 0): z_egj + half_width + 1].max(axis=1)
    return float(np.sort(sleeve)[: int(4 * fs)].mean())


def transition_zone(p: np.ndarray, t_ues: float, z_egj: int, fs: float = FS) -> int:
    """Pressure trough between proximal and distal contraction: lowest peak pressure in the upper body."""
    win = p[int(t_ues * fs): int((t_ues + 12) * fs)]
    top, bottom = 4, max(5, (z_egj - 4) // 2)
    return int(top + np.argmin(win[:, top: bottom + 1].max(axis=0)))


def dci(p: np.ndarray, t_ues: float, z_tz: int, z_egj: int, fs: float = FS, dz: float = 1.0) -> float:
    """Distal contractile integral (mmHg*s*cm) over the 15 s after UES relaxation."""
    seg = p[int(t_ues * fs): int((t_ues + 15) * fs), z_tz + 1: z_egj - 2]
    return float(np.clip(seg - 20, 0, None).sum() / fs * dz)


def distal_latency(p: np.ndarray, t_ues: float, z_tz: int, z_egj: int, fs: float = FS) -> float:
    """DL (s): onset of the 30 mmHg contour along the distal body, fitted with two straight lines."""
    zs, ts = [], []
    i0 = int((t_ues + 0.5) * fs)
    for z in range(z_tz + 1, z_egj - 1):
        hit = np.flatnonzero(p[i0: i0 + int(15 * fs), z] >= 30)
        if hit.size:
            zs.append(z)
            ts.append(t_ues + 0.5 + hit[0] / fs)
    if len(zs) < 6:
        return float("nan")
    zs, ts = np.array(zs, float), np.array(ts)
    best = (np.inf, None)
    for k in range(2, len(zs) - 2):           # continuous two-segment fit, breakpoint at zs[k]
        zb = zs[k]
        a = np.column_stack([np.ones_like(zs), np.minimum(zs - zb, 0), np.maximum(zs - zb, 0)])
        coef, *_ = np.linalg.lstsq(a, ts, rcond=None)
        sse = float(((a @ coef - ts) ** 2).sum())
        if sse < best[0]:
            best = (sse, coef[0])
    return float(best[1] - t_ues)


def largest_break(p: np.ndarray, t_ues: float, z_egj: int, fs: float = FS, top: int = 3) -> float:
    """Longest run (cm) of body sensors that never reach 20 mmHg after the swallow."""
    win = p[int(t_ues * fs): int((t_ues + 15) * fs), top: z_egj - 2]
    quiet = win.max(axis=0) < 20
    best = run = 0
    for q in quiet:
        run = run + 1 if q else 0
        best = max(best, run)
    return float(best)


def panesophageal(p: np.ndarray, t_ues: float, z_egj: int, fs: float = FS, top: int = 3,
                  level: float = 30.0, min_s: float = 1.0) -> bool:
    """True if the whole body (top .."""
    win = p[int(t_ues * fs): int((t_ues + 15) * fs), top: z_egj - 1]
    allp = (win >= level).all(axis=1)
    run = best = 0
    for a in allp:
        run = run + 1 if a else 0
        best = max(best, run)
    return best / fs >= min_s


def analyse_swallow(p: np.ndarray, fs: float = FS) -> SwallowMetrics:
    """All CCv4 metrics of one swallow."""
    t_ues = ues_relaxation(p, fs)
    z_egj = egj_location(p, t_ues, fs)
    z_tz = transition_zone(p, t_ues, z_egj, fs)
    return SwallowMetrics(irp=irp(p, t_ues, z_egj, fs), dci=dci(p, t_ues, z_tz, z_egj, fs),
                          dl=distal_latency(p, t_ues, z_tz, z_egj, fs), break_cm=largest_break(p, t_ues, z_egj, fs),
                          pan=panesophageal(p, t_ues, z_egj, fs), t_ues=t_ues, z_egj=z_egj, z_tz=z_tz)
