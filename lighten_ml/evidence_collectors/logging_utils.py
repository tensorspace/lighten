"""
Logging utilities for LLM-based evidence extraction.

This module provides helper functions and utilities for comprehensive logging
of the extraction pipeline, including performance metrics, confidence analysis,
and decision reasoning.
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ExtractionLogger:
    """
    Utility class for structured logging of evidence extraction processes.

    Provides consistent logging patterns across all extraction methods with
    detailed performance metrics, confidence analysis, and decision reasoning.
    """

    def __init__(self, component_name: str):
        """
        Initialize extraction logger for a specific component.

        Args:
            component_name: Name of the extraction component (e.g., 'LLM-CLINICAL', 'HYBRID')
        """
        self.component_name = component_name
        self.logger = logging.getLogger(f"{__name__}.{component_name}")

    def log_extraction_start(
        self, patient_id: str, hadm_id: str, method: str, **kwargs
    ):
        """
        Log the start of an extraction process.

        Args:
            patient_id: Patient identifier
            hadm_id: Hospital admission identifier
            method: Extraction method being used
            **kwargs: Additional parameters to log
        """
        self.logger.info(
            f"[{self.component_name}] Starting {method} extraction for patient {patient_id}, admission {hadm_id}"
        )

        if kwargs:
            params_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
            self.logger.debug(f"[{self.component_name}] Parameters: {params_str}")

    def log_extraction_complete(self, processing_time: float, result: Dict[str, Any]):
        """
        Log the completion of an extraction process with results summary.

        Args:
            processing_time: Total processing time in seconds
            result: Extraction result dictionary
        """
        symptoms_count = len(result.get("symptoms", []))
        findings_count = len(result.get("ecg_findings", []))
        method_used = result.get("extraction_method", "unknown")

        self.logger.info(
            f"[{self.component_name}] Extraction completed in {processing_time:.2f}s using {method_used}"
        )
        self.logger.info(
            f"[{self.component_name}] Results: {symptoms_count} symptoms, {findings_count} ECG findings"
        )

        # Log confidence statistics
        confidence_scores = result.get("confidence_scores", {})
        if confidence_scores:
            avg_confidence = sum(confidence_scores.values()) / len(confidence_scores)
            max_confidence = max(confidence_scores.values())
            min_confidence = min(confidence_scores.values())

            self.logger.info(
                f"[{self.component_name}] Confidence stats - avg: {avg_confidence:.3f}, "
                f"max: {max_confidence:.3f}, min: {min_confidence:.3f}"
            )

    def log_note_processing(
        self,
        note_idx: int,
        total_notes: int,
        note_type: str,
        chart_date: datetime,
        processing_time: float,
        results_count: int,
    ):
        """
        Log the processing of individual clinical notes.

        Args:
            note_idx: Index of current note
            total_notes: Total number of notes to process
            note_type: Type of clinical note
            chart_date: Date of the note
            processing_time: Time taken to process this note
            results_count: Number of results extracted from this note
        """
        self.logger.debug(
            f"[{self.component_name}] Processing note {note_idx}/{total_notes} "
            f"from {chart_date} (type: {note_type})"
        )
        self.logger.debug(
            f"[{self.component_name}] Note {note_idx} completed in {processing_time:.2f}s "
            f"with {results_count} results"
        )

    def log_llm_interaction(
        self,
        prompt_length: int,
        response_time: float,
        response_valid: bool,
        tokens_used: Optional[int] = None,
    ):
        """
        Log LLM API interactions with performance metrics.

        Args:
            prompt_length: Length of prompt sent to LLM
            response_time: Time taken for LLM response
            response_valid: Whether response was valid/parseable
            tokens_used: Number of tokens used (if available)
        """
        self.logger.debug(
            f"[{self.component_name}] LLM request - prompt: {prompt_length} chars, "
            f"response: {response_time:.2f}s, valid: {response_valid}"
        )

        if tokens_used:
            self.logger.debug(f"[{self.component_name}] LLM tokens used: {tokens_used}")

    def log_confidence_analysis(
        self, confidence_scores: Dict[str, float], threshold: float, decision: str
    ):
        """
        Log confidence score analysis and threshold decisions.

        Args:
            confidence_scores: Dictionary of confidence scores
            threshold: Confidence threshold used
            decision: Decision made based on confidence analysis
        """
        if not confidence_scores:
            self.logger.debug(
                f"[{self.component_name}] No confidence scores available for analysis"
            )
            return

        avg_confidence = sum(confidence_scores.values()) / len(confidence_scores)
        high_confidence_count = sum(
            1 for score in confidence_scores.values() if score > threshold
        )

        self.logger.info(
            f"[{self.component_name}] Confidence analysis - avg: {avg_confidence:.3f}, "
            f"threshold: {threshold}, above_threshold: {high_confidence_count}/{len(confidence_scores)}"
        )
        self.logger.info(f"[{self.component_name}] Decision: {decision}")

    def log_method_selection(
        self,
        requested_method: Optional[str],
        selected_method: str,
        reason: str,
        available_methods: List[str],
    ):
        """
        Log method selection reasoning for hybrid extraction.

        Args:
            requested_method: Method requested by user
            selected_method: Method actually selected
            reason: Reason for selection
            available_methods: List of available methods
        """
        self.logger.info(
            f"[{self.component_name}] Method selection - requested: {requested_method}, "
            f"selected: {selected_method}"
        )
        self.logger.debug(
            f"[{self.component_name}] Available methods: {available_methods}"
        )
        self.logger.info(f"[{self.component_name}] Selection reason: {reason}")

    def log_error_with_context(self, error: Exception, context: Dict[str, Any]):
        """
        Log errors with detailed context information.

        Args:
            error: Exception that occurred
            context: Context information (patient_id, note_type, etc.)
        """
        context_str = ", ".join([f"{k}={v}" for k, v in context.items()])
        self.logger.error(f"[{self.component_name}] Error: {str(error)}")
        self.logger.debug(f"[{self.component_name}] Error context: {context_str}")

    def log_performance_metrics(self, metrics: Dict[str, Any]):
        """
        Log detailed performance metrics.

        Args:
            metrics: Dictionary containing performance metrics
        """
        self.logger.info(f"[{self.component_name}] Performance metrics:")

        for category, values in metrics.items():
            if isinstance(values, dict):
                for key, value in values.items():
                    self.logger.info(
                        f"[{self.component_name}]   {category}.{key}: {value}"
                    )
            else:
                self.logger.info(f"[{self.component_name}]   {category}: {values}")

    def log_deduplication_stats(
        self, raw_count: int, final_count: int, duplicates_removed: int, category: str
    ):
        """
        Log deduplication statistics.

        Args:
            raw_count: Number of raw items before deduplication
            final_count: Number of items after deduplication
            duplicates_removed: Number of duplicates removed
            category: Category of items (symptoms, findings, etc.)
        """
        dedup_rate = (duplicates_removed / raw_count * 100) if raw_count > 0 else 0

        self.logger.info(
            f"[{self.component_name}] {category} deduplication: "
            f"{raw_count} → {final_count} ({duplicates_removed} removed, {dedup_rate:.1f}%)"
        )

    def log_high_value_findings(
        self, findings: List[Dict[str, Any]], confidence_threshold: float = 0.8
    ):
        """
        Log high-confidence or clinically significant findings.

        Args:
            findings: List of findings to analyze
            confidence_threshold: Minimum confidence for "high-value" classification
        """
        high_value_findings = [
            f
            for f in findings
            if f.get("confidence", 0) > confidence_threshold
            or f.get("mi_related", False)
        ]

        if high_value_findings:
            finding_names = [
                f.get("name", f.get("finding", "unknown")) for f in high_value_findings
            ]
            self.logger.info(
                f"[{self.component_name}] High-value findings ({len(high_value_findings)}): "
                f"{finding_names}"
            )
        else:
            self.logger.debug(
                f"[{self.component_name}] No high-value findings identified"
            )


def setup_extraction_logging(
    log_level: str = "INFO", log_format: Optional[str] = None
) -> logging.Logger:
    """
    Set up comprehensive logging for extraction pipeline.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_format: Custom log format string

    Returns:
        Configured logger instance
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Configure root logger for extraction
    extraction_logger = logging.getLogger("lighten_ml.evidence_collectors")
    extraction_logger.setLevel(getattr(logging, log_level.upper()))

    # Create console handler if not exists
    if not extraction_logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))

        formatter = logging.Formatter(log_format)
        console_handler.setFormatter(formatter)

        extraction_logger.addHandler(console_handler)

    return extraction_logger


def log_extraction_summary(results: Dict[str, Any], component: str = "EXTRACTION"):
    """
    Log a comprehensive summary of extraction results.

    Args:
        results: Complete extraction results
        component: Component name for logging prefix
    """
    logger = logging.getLogger(__name__)

    logger.info(f"[{component}] === EXTRACTION SUMMARY ===")

    # Basic counts
    symptoms = results.get("symptoms", [])
    ecg_findings = results.get("ecg_findings", [])
    negated_symptoms = results.get("negated_symptoms", [])

    logger.info(f"[{component}] Total symptoms extracted: {len(symptoms)}")
    logger.info(f"[{component}] Total ECG findings: {len(ecg_findings)}")
    logger.info(f"[{component}] Negated symptoms: {len(negated_symptoms)}")

    # Method and performance
    method = results.get("extraction_method", "unknown")
    extraction_details = results.get("extraction_details", {})

    logger.info(f"[{component}] Extraction method: {method}")

    if "processing_time" in extraction_details:
        logger.info(
            f"[{component}] Processing time: {extraction_details['processing_time']:.2f}s"
        )

    # Confidence analysis
    confidence_scores = results.get("confidence_scores", {})
    if confidence_scores:
        avg_confidence = sum(confidence_scores.values()) / len(confidence_scores)
        logger.info(f"[{component}] Average confidence: {avg_confidence:.3f}")

    # High-confidence findings
    high_conf_symptoms = [s for s in symptoms if s.get("confidence", 0) > 0.8]
    mi_related_findings = [f for f in ecg_findings if f.get("mi_related", False)]

    if high_conf_symptoms:
        logger.info(
            f"[{component}] High-confidence symptoms: {[s.get('name') for s in high_conf_symptoms]}"
        )

    if mi_related_findings:
        logger.info(
            f"[{component}] MI-related ECG findings: {[f.get('finding') for f in mi_related_findings]}"
        )

    logger.info(f"[{component}] === END SUMMARY ===")
