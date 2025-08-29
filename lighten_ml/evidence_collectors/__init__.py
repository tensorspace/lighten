"""Evidence collection modules for clinical data analysis."""

from .base_evidence_collector import BaseEvidenceCollector
from .troponin_analyzer import TroponinAnalyzer
from .clinical_evidence_extractor import ClinicalEvidenceExtractor
from .ecg_evidence_extractor import ECGEvidenceExtractor
from .imaging_evidence_extractor import ImagingEvidenceExtractor
from .angiography_evidence_extractor import AngiographyEvidenceExtractor

__all__ = [
    'BaseEvidenceCollector',
    'TroponinAnalyzer',
    'ClinicalEvidenceExtractor',
    'ECGEvidenceExtractor',
    'ImagingEvidenceExtractor',
    'AngiographyEvidenceExtractor'
]
