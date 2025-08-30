"""
Hybrid evidence extractor that can switch between regex and LLM-based extraction methods.

This module provides a configurable approach to clinical text extraction, allowing
users to choose between traditional regex patterns and advanced LLM-based extraction
based on their needs, resources, and accuracy requirements.
"""

import logging
import time
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from ..llm_client import LightenLLMClient
from .clinical_evidence_extractor import ClinicalEvidenceExtractor
from .ecg_evidence_extractor import ECGEvidenceExtractor
from .llm_clinical_evidence_extractor import LLMClinicalEvidenceExtractor
from .llm_ecg_evidence_extractor import LLMECGEvidenceExtractor

logger = logging.getLogger(__name__)


class ExtractionMethod(Enum):
    """Enumeration of available extraction methods."""

    REGEX = "regex"
    LLM = "llm"
    HYBRID = "hybrid"


class HybridEvidenceExtractor:
    """
    Hybrid evidence extractor that can use different extraction methods.

    Provides flexibility to choose between:
    - Regex-based extraction (fast, deterministic, limited accuracy)
    - LLM-based extraction (slower, context-aware, high accuracy)
    - Hybrid approach (LLM with regex fallback)
    """

    def __init__(
        self,
        notes_data_loader,
        llm_client: Optional[LightenLLMClient] = None,
        max_notes: Optional[int] = None,
        default_method: ExtractionMethod = ExtractionMethod.HYBRID,
        confidence_threshold: float = 0.7,
    ):
        """
        Initialize the hybrid evidence extractor.

        Args:
            notes_data_loader: Data loader for clinical notes
            llm_client: LLM client for advanced extraction (optional)
            max_notes: Maximum number of notes to process per admission
            default_method: Default extraction method to use
            confidence_threshold: Minimum confidence for LLM results before fallback
        """
        self.notes_data_loader = notes_data_loader
        self.llm_client = llm_client
        self.max_notes = max_notes
        self.default_method = default_method
        self.confidence_threshold = confidence_threshold

        # Initialize extractors
        self._init_extractors()

    def _init_extractors(self):
        """Initialize all available extractors."""
        # Regex-based extractors (always available)
        self.regex_clinical = ClinicalEvidenceExtractor(
            notes_data_loader=self.notes_data_loader,
            llm_client=None,  # Regex doesn't need LLM
            max_notes=self.max_notes,
        )

        self.regex_ecg = ECGEvidenceExtractor(
            notes_data_loader=self.notes_data_loader,
            llm_client=None,  # Regex doesn't need LLM
            max_notes=self.max_notes,
        )

        # LLM-based extractors (only if LLM client available)
        if self.llm_client:
            self.llm_clinical = LLMClinicalEvidenceExtractor(
                notes_data_loader=self.notes_data_loader,
                llm_client=self.llm_client,
                max_notes=self.max_notes,
            )

            self.llm_ecg = LLMECGEvidenceExtractor(
                notes_data_loader=self.notes_data_loader,
                llm_client=self.llm_client,
                max_notes=self.max_notes,
            )
        else:
            self.llm_clinical = None
            self.llm_ecg = None
            logger.warning("No LLM client provided - LLM-based extraction unavailable")

    def collect_clinical_evidence(
        self, patient_id: str, hadm_id: str, method: Optional[ExtractionMethod] = None
    ) -> Dict[str, Any]:
        """
        Collect clinical evidence using specified or default method.

        Args:
            patient_id: The ID of the patient
            hadm_id: The ID of the hospital admission
            method: Extraction method to use (overrides default)

        Returns:
            Dictionary containing clinical evidence
        """
        extraction_method = method or self.default_method

        logger.info(
            f"Collecting clinical evidence for patient {patient_id}, "
            f"admission {hadm_id} using {extraction_method.value} method"
        )

        if extraction_method == ExtractionMethod.REGEX:
            return self._extract_clinical_regex(patient_id, hadm_id)

        elif extraction_method == ExtractionMethod.LLM:
            return self._extract_clinical_llm(patient_id, hadm_id)

        elif extraction_method == ExtractionMethod.HYBRID:
            return self._extract_clinical_hybrid(patient_id, hadm_id)

        else:
            logger.error(f"Unknown extraction method: {extraction_method}")
            return self._extract_clinical_regex(patient_id, hadm_id)  # Fallback

    def collect_ecg_evidence(
        self, patient_id: str, hadm_id: str, method: Optional[ExtractionMethod] = None
    ) -> Dict[str, Any]:
        """
        Collect ECG evidence using specified or default method.

        Args:
            patient_id: The ID of the patient
            hadm_id: The ID of the hospital admission
            method: Extraction method to use (overrides default)

        Returns:
            Dictionary containing ECG evidence
        """
        extraction_method = method or self.default_method

        logger.info(
            f"Collecting ECG evidence for patient {patient_id}, "
            f"admission {hadm_id} using {extraction_method.value} method"
        )

        if extraction_method == ExtractionMethod.REGEX:
            return self._extract_ecg_regex(patient_id, hadm_id)

        elif extraction_method == ExtractionMethod.LLM:
            return self._extract_ecg_llm(patient_id, hadm_id)

        elif extraction_method == ExtractionMethod.HYBRID:
            return self._extract_ecg_hybrid(patient_id, hadm_id)

        else:
            logger.error(f"Unknown extraction method: {extraction_method}")
            return self._extract_ecg_regex(patient_id, hadm_id)  # Fallback

    def _extract_clinical_regex(self, patient_id: str, hadm_id: str) -> Dict[str, Any]:
        """Extract clinical evidence using regex patterns."""
        return self.regex_clinical.collect_evidence(patient_id, hadm_id)

    def _extract_clinical_llm(self, patient_id: str, hadm_id: str) -> Dict[str, Any]:
        """Extract clinical evidence using LLM."""
        if not self.llm_clinical:
            logger.warning(
                "LLM clinical extractor not available, falling back to regex"
            )
            return self._extract_clinical_regex(patient_id, hadm_id)

        return self.llm_clinical.collect_evidence(patient_id, hadm_id)

    def _extract_clinical_hybrid(self, patient_id: str, hadm_id: str) -> Dict[str, Any]:
        """Extract clinical evidence using hybrid approach (LLM with regex fallback)."""
        if not self.llm_clinical:
            logger.info("LLM not available, using regex for clinical evidence")
            return self._extract_clinical_regex(patient_id, hadm_id)

        try:
            # Try LLM extraction first
            llm_result = self.llm_clinical.collect_evidence(patient_id, hadm_id)

            # Check if LLM result meets confidence threshold
            if self._meets_confidence_threshold(llm_result):
                logger.info("LLM extraction successful with high confidence")
                llm_result["extraction_method"] = "llm_primary"
                return llm_result
            else:
                logger.info(
                    "LLM extraction below confidence threshold, using hybrid approach"
                )

        except Exception as e:
            logger.error(f"LLM extraction failed: {str(e)}, falling back to regex")

        # Fallback to regex or combine results
        regex_result = self._extract_clinical_regex(patient_id, hadm_id)

        # If we have partial LLM results, combine them
        try:
            if "llm_result" in locals() and llm_result:
                combined_result = self._combine_clinical_results(
                    llm_result, regex_result
                )
                combined_result["extraction_method"] = "hybrid_combined"
                return combined_result
        except:
            pass

        # Pure regex fallback
        regex_result["extraction_method"] = "regex_fallback"
        return regex_result

    def _extract_ecg_regex(self, patient_id: str, hadm_id: str) -> Dict[str, Any]:
        """Extract ECG evidence using regex patterns."""
        return self.regex_ecg.collect_evidence(patient_id, hadm_id)

    def _extract_ecg_llm(self, patient_id: str, hadm_id: str) -> Dict[str, Any]:
        """Extract ECG evidence using LLM."""
        if not self.llm_ecg:
            logger.warning("LLM ECG extractor not available, falling back to regex")
            return self._extract_ecg_regex(patient_id, hadm_id)

        return self.llm_ecg.collect_evidence(patient_id, hadm_id)

    def _extract_ecg_hybrid(self, patient_id: str, hadm_id: str) -> Dict[str, Any]:
        """Extract ECG evidence using hybrid approach (LLM with regex fallback)."""
        if not self.llm_ecg:
            logger.info("LLM not available, using regex for ECG evidence")
            return self._extract_ecg_regex(patient_id, hadm_id)

        try:
            # Try LLM extraction first
            llm_result = self.llm_ecg.collect_evidence(patient_id, hadm_id)

            # Check if LLM result meets confidence threshold
            if self._meets_confidence_threshold(llm_result):
                logger.info("LLM ECG extraction successful with high confidence")
                llm_result["extraction_method"] = "llm_primary"
                return llm_result
            else:
                logger.info(
                    "LLM ECG extraction below confidence threshold, using hybrid approach"
                )

        except Exception as e:
            logger.error(f"LLM ECG extraction failed: {str(e)}, falling back to regex")

        # Fallback to regex or combine results
        regex_result = self._extract_ecg_regex(patient_id, hadm_id)

        # If we have partial LLM results, combine them
        try:
            if "llm_result" in locals() and llm_result:
                combined_result = self._combine_ecg_results(llm_result, regex_result)
                combined_result["extraction_method"] = "hybrid_combined"
                return combined_result
        except:
            pass

        # Pure regex fallback
        regex_result["extraction_method"] = "regex_fallback"
        return regex_result

    def _meets_confidence_threshold(self, result: Dict[str, Any]) -> bool:
        """
        Check if extraction result meets minimum confidence threshold.

        Args:
            result: Extraction result dictionary

        Returns:
            True if result meets confidence threshold
        """
        confidence_scores = result.get("confidence_scores", {})

        if not confidence_scores:
            return False

        # Calculate average confidence
        avg_confidence = sum(confidence_scores.values()) / len(confidence_scores)

        return avg_confidence >= self.confidence_threshold

    def _combine_clinical_results(
        self, llm_result: Dict[str, Any], regex_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Combine LLM and regex clinical extraction results.

        Args:
            llm_result: LLM extraction result
            regex_result: Regex extraction result

        Returns:
            Combined result dictionary
        """
        combined = regex_result.copy()

        # Merge symptoms, prioritizing high-confidence LLM results
        llm_symptoms = llm_result.get("symptoms", [])
        regex_symptoms = regex_result.get("symptoms", [])

        # Add high-confidence LLM symptoms
        for symptom in llm_symptoms:
            if symptom.get("confidence", 0) >= self.confidence_threshold:
                combined["symptoms"].append(symptom)

        # Add LLM onset dates if available
        if llm_result.get("symptom_onset_dates"):
            combined["symptom_onset_dates"] = llm_result["symptom_onset_dates"]

        # Merge confidence scores
        combined["confidence_scores"] = {
            **regex_result.get("confidence_scores", {}),
            **llm_result.get("confidence_scores", {}),
        }

        return combined

    def _combine_ecg_results(
        self, llm_result: Dict[str, Any], regex_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Combine LLM and regex ECG extraction results.

        Args:
            llm_result: LLM extraction result
            regex_result: Regex extraction result

        Returns:
            Combined result dictionary
        """
        combined = regex_result.copy()

        # Merge ECG findings, prioritizing high-confidence LLM results
        llm_findings = llm_result.get("ecg_findings", [])

        # Add high-confidence LLM findings
        for finding in llm_findings:
            if finding.get("confidence", 0) >= self.confidence_threshold:
                combined["ecg_findings"].append(finding)

        # Add LLM-specific data if available
        if llm_result.get("lead_specific_findings"):
            combined["lead_specific_findings"] = llm_result["lead_specific_findings"]

        if llm_result.get("temporal_context"):
            combined["temporal_context"] = llm_result["temporal_context"]

        # Merge confidence scores
        combined["confidence_scores"] = {
            **regex_result.get("confidence_scores", {}),
            **llm_result.get("confidence_scores", {}),
        }

        return combined

    def get_available_methods(self) -> List[ExtractionMethod]:
        """
        Get list of available extraction methods based on configuration.

        Returns:
            List of available extraction methods
        """
        methods = [ExtractionMethod.REGEX]  # Always available

        if self.llm_client:
            methods.extend([ExtractionMethod.LLM, ExtractionMethod.HYBRID])

        return methods

    def set_default_method(self, method: ExtractionMethod):
        """
        Set the default extraction method.

        Args:
            method: New default extraction method
        """
        if method not in self.get_available_methods():
            raise ValueError(f"Method {method.value} is not available")

        self.default_method = method
        logger.info(f"Default extraction method set to: {method.value}")

    def set_confidence_threshold(self, threshold: float):
        """
        Set the confidence threshold for hybrid extraction.

        Args:
            threshold: New confidence threshold (0.0-1.0)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")

        self.confidence_threshold = threshold
        logger.info(f"Confidence threshold set to: {threshold}")
