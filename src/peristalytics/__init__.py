"""peristalytics: synthetic high-resolution oesophageal manometry and Chicago Classification v4.0 analysis."""
from .classify import diagnose, diagnose_study
from .metrics import SwallowMetrics, analyse_swallow
from .synth import PHENOTYPES, TYPES, SwallowSpec, study, swallow

__all__ = ["swallow", "study", "SwallowSpec", "TYPES", "PHENOTYPES", "analyse_swallow", "SwallowMetrics",
           "diagnose", "diagnose_study"]
__version__ = "0.1.0"
