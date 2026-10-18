"""The workflow's step pages: Load › Calibrate › Detect › Review › Measure › Gate › Report › Batch."""

from .batch import BatchStep
from .calibrate import CalibrateStep
from .detect import DetectStep
from .gate import GateStep
from .load import LoadStep
from .measure import MeasureStep
from .report import ReportStep
from .review import ReviewStep

PAGES = {
    "load": LoadStep,
    "calibrate": CalibrateStep,
    "detect": DetectStep,
    "review": ReviewStep,
    "measure": MeasureStep,
    "gate": GateStep,
    "report": ReportStep,
    "batch": BatchStep,
}

__all__ = [
    "PAGES",
    "BatchStep",
    "CalibrateStep",
    "DetectStep",
    "GateStep",
    "LoadStep",
    "MeasureStep",
    "ReportStep",
    "ReviewStep",
]
