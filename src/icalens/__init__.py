"""Public package interface for ICA Lens."""

from .analysis import AnalysisResult, CaptureResult
from .exceptions import ArtifactError, ICALensError, NotFittedError
from .lens import ICALens

__all__ = [
    "AnalysisResult",
    "ArtifactError",
    "CaptureResult",
    "ICALens",
    "ICALensError",
    "NotFittedError",
]
__version__ = "0.3.0.dev1"
