"""Chicago Classification v4.0 study diagnosis from ten supine swallows."""
from __future__ import annotations

import numpy as np

from .metrics import SwallowMetrics, analyse_swallow

IRP_ULN_SUPINE = 15.0


def diagnose(metrics: list[SwallowMetrics]) -> dict:
    n = len(metrics)
    types = [m.swallow_type for m in metrics]
    frac = {k: types.count(k) / n for k in ("failed", "weak", "fragmented", "premature", "hypercontractile",
                                             "normal")}
    pan = sum(m.pan for m in metrics) / n
    ineffective = frac["failed"] + frac["weak"] + frac["fragmented"]
    med_irp = float(np.median([m.irp for m in metrics]))
    peristalsis = frac["normal"] + frac["weak"] + frac["fragmented"] + frac["hypercontractile"] > 0
    if med_irp > IRP_ULN_SUPINE:
        if frac["failed"] == 1.0 and pan >= 0.2:
            dx = "achalasia II"
        elif frac["failed"] == 1.0:
            dx = "achalasia I"
        elif frac["premature"] >= 0.2 and not peristalsis:
            dx = "achalasia III"
        else:
            dx = "EGJ outflow obstruction"
    elif frac["failed"] == 1.0:
        dx = "absent contractility"
    elif frac["premature"] >= 0.2:
        dx = "distal oesophageal spasm"
    elif frac["hypercontractile"] >= 0.2:
        dx = "hypercontractile oesophagus"
    elif ineffective > 0.7 or frac["failed"] >= 0.5:
        dx = "ineffective oesophageal motility"
    else:
        dx = "normal"
    return {"diagnosis": dx, "median_irp": med_irp, "swallow_types": types, "fractions": frac,
            "pan_fraction": pan, "ineffective_fraction": ineffective}


def diagnose_study(swallows) -> dict:
    """Analyse raw swallows P[t, z] and return the diagnosis plus per-swallow metrics."""
    m = [analyse_swallow(p) for p in swallows]
    out = diagnose(m)
    out["metrics"] = m
    return out
