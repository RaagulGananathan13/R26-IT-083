"""Component 02 — XAI-based ECG abnormality detection and cardiac risk reporting.

Import order matters only in that `models` defines the shared constants.
"""
from .models import CLASS_NAMES, LEAD_NAMES, NUM_CLASSES, SAMPLING_RATE, SIGNAL_LENGTH
from .models import ECGResNet, ECGResNetSE, FocalLoss, build_model
from .models import (DEFAULT_MODEL, MODEL_REGISTRY, ModelSpec, describe_models,
                     get_spec, list_models, register_model, resolve_model_name)
from .calibration import TemperatureCalibrator, calibration_report
from .conformal import ConformalTriage, RULE_IN, RULE_OUT, REFER, risk_coverage_curve
from .pipeline import ECGPipeline, AnalysisResult
from .zoo import ConsensusResult, ModelAssets, ModelBundle, ModelZoo
from .quality import QualityReport
from .report import ClinicalReport, build_report
from .verify import verify_report, verify_paraphrase, safe_paraphrase, batch_verify

__all__ = [
    "CLASS_NAMES", "LEAD_NAMES", "NUM_CLASSES", "SAMPLING_RATE", "SIGNAL_LENGTH",
    "ECGResNet", "ECGResNetSE", "FocalLoss", "build_model",
    "DEFAULT_MODEL", "MODEL_REGISTRY", "ModelSpec", "describe_models",
    "get_spec", "list_models", "register_model", "resolve_model_name",
    "TemperatureCalibrator", "calibration_report",
    "ConformalTriage", "RULE_IN", "RULE_OUT", "REFER", "risk_coverage_curve",
    "ECGPipeline", "AnalysisResult", "QualityReport",
    "ModelZoo", "ModelBundle", "ModelAssets", "ConsensusResult",
    "ClinicalReport", "build_report",
    "verify_report", "verify_paraphrase", "safe_paraphrase", "batch_verify",
]
__version__ = "0.2.0"
