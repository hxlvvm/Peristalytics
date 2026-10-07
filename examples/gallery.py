"""Phenotype gallery and robustness benchmark (writes assets/gallery.png, assets/robustness.png)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from peristalsim import PHENOTYPES, SwallowSpec, TYPES, diagnose_study, study, swallow
from peristalsim.synth import DURATION, N_SENSORS

Path("assets").mkdir(exist_ok=True)

# ---- 1. gallery: one representative swallow per phenotype, with the computed diagnosis
show = {"normal": "normal", "achalasia I": "failed", "achalasia II": "pan", "achalasia III": "spastic",
        "EGJ outflow obstruction": "normal", "absent contractility": "failed",
        "distal oesophageal spasm": "premature", "hypercontractile oesophagus": "hypercontractile",
        "ineffective oesophageal motility": "weak"}
fig, axs = plt.subplots(3, 3, figsize=(12, 9), sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.42, wspace=0.12)
for ax, (ph, typ) in zip(axs.ravel(), show.items()):
    swallows, truth = study(ph, seed=0)
    dx = diagnose_study(swallows)
    p = swallows[truth.index(typ)]
    im = ax.imshow(p.T, aspect="auto", cmap="jet", vmin=-5, vmax=150, extent=[0, DURATION, N_SENSORS, 0])
    verdict = "correct" if dx["diagnosis"] == ph else dx["diagnosis"]
    ax.set_title(f"{ph}\ncomputed: {verdict} (IRP {dx['median_irp']:.0f} mmHg)", fontsize=8.5)
for ax in axs[-1]:
    ax.set_xlabel("time (s)", fontsize=8)
for ax in axs[:, 0]:
    ax.set_ylabel("position (cm)", fontsize=8)
fig.colorbar(im, ax=axs, shrink=0.6, label="pressure (mmHg)")
fig.suptitle("Synthetic high-resolution manometry and its Chicago Classification v4.0 diagnosis", fontsize=11)
fig.savefig("assets/gallery.png", dpi=110, bbox_inches="tight")

# ---- 2. robustness: diagnostic accuracy as noise and swallow-to-swallow variability grow
levels = [(1.5, 0.15), (3.0, 0.25), (5.0, 0.35), (8.0, 0.45)]
phen = list(PHENOTYPES)
n = 20
acc = np.zeros((len(phen), len(levels)))
import peristalsim.synth as synth
orig = synth.swallow
for j, (noise, jitter) in enumerate(levels):
    def noisy(spec, rng=None, _noise=noise):
        return orig(SwallowSpec(**{**spec.__dict__, "noise": _noise}), rng)
    synth.swallow = noisy
    for i, ph in enumerate(phen):
        acc[i, j] = np.mean([diagnose_study(synth.study(ph, seed=s, jitter=jitter)[0])["diagnosis"] == ph
                             for s in range(n)])
synth.swallow = orig
print(f"{'phenotype':34s}" + "".join(f"  noise {a:.1f}/jit {b:.2f}" for a, b in levels))
for i, ph in enumerate(phen):
    print(f"{ph:34s}" + "".join(f"{100 * acc[i, j]:20.0f}%" for j in range(len(levels))))
print(f"{'mean':34s}" + "".join(f"{100 * acc[:, j].mean():20.0f}%" for j in range(len(levels))))

fig, ax = plt.subplots(figsize=(7.5, 4.6))
im = ax.imshow(acc * 100, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(len(levels)), [f"noise {a:.1f} mmHg\njitter {int(b * 100)} %" for a, b in levels], fontsize=8)
ax.set_yticks(range(len(phen)), phen, fontsize=8)
for i in range(len(phen)):
    for j in range(len(levels)):
        ax.text(j, i, f"{acc[i, j] * 100:.0f}", ha="center", va="center", fontsize=8)
ax.set_title(f"Diagnostic accuracy on {n} synthetic studies per cell", fontsize=10)
fig.colorbar(im, label="% correct")
fig.tight_layout()
fig.savefig("assets/robustness.png", dpi=110)
print("wrote assets/gallery.png, assets/robustness.png")
