"""Evidence extractor for imaging findings related to MI."""

import logging
import re
from typing import Any, Dict, List, Optional

from .base_evidence_collector import BaseEvidenceCollector

logger = logging.getLogger(__name__)


class ImagingEvidenceExtractor(BaseEvidenceCollector):
    """Extracts imaging evidence (e.g., from echocardiograms) for MI."""

    IMAGING_PATTERNS = [
        {
            "name": "Wall Motion Abnormality",
            "pattern": re.compile(
                r"(wall motion abnormalit(y|ies)|WMA|hypokinesis|akinesis|dyskinesis)",
                re.IGNORECASE,
            ),
            "mi_related": True,
        },
        {
            "name": "Loss of Viable Myocardium",
            "pattern": re.compile(
                r"(loss of.*viab|non-viable|infarct|scar|late gadolinium enhancement|lge)",
                re.IGNORECASE,
            ),
            "mi_related": True,
        },
    ]

    def __init__(
        self,
        notes_data_loader: Any,
        llm_client: Optional[Any] = None,
        max_notes: Optional[int] = None,
    ):
        """Initialize the imaging evidence extractor.

        Args:
            notes_data_loader: Instance of ClinicalNotesDataLoader
            llm_client: Optional instance of the LLM client
            max_notes: Maximum number of notes to process
        """
        super().__init__(
            notes_data_loader=notes_data_loader,
            llm_client=llm_client,
            max_notes=max_notes,
        )

    def collect_evidence(self, patient_id: str, hadm_id: str) -> Dict[str, Any]:
        """Collect imaging evidence from notes for a specific admission."""
        log_prefix = f"[{patient_id}][{hadm_id}] [IMAGING_EXTRACTOR]"

        evidence = self._get_evidence_base()
        evidence["imaging_findings"] = []
        evidence["metadata"] = {"extraction_mode": "none"}

        logger.info(f"{log_prefix} Fetching clinical notes for imaging evidence.")
        notes = self.notes_data_loader.get_patient_notes(patient_id, hadm_id)
        # Filter for relevant notes like echo reports
        imaging_notes = notes[notes["note_type"].isin(["Echo", "Radiology"])]
        logger.info(f"{log_prefix} Found {len(imaging_notes)} imaging reports out of {len(notes)} total notes.")

        # Limit number of notes if configured
        if self.max_notes and len(imaging_notes) > self.max_notes:
            logger.info(f"{log_prefix} Limiting notes from {len(imaging_notes)} to {self.max_notes}.")
            imaging_notes = imaging_notes.head(self.max_notes)

        if imaging_notes.empty:
            logger.warning(f"{log_prefix} No imaging reports found.")
            return evidence

        logger.info(f"{log_prefix} Processing {len(imaging_notes)} imaging reports.")

        try_llm = self.llm_client and self.llm_client.enabled
        logger.info(f"{log_prefix} Method selection: LLM available={try_llm}")

        extracted_with_llm = False
        if try_llm:
            try:
                logger.info(f"{log_prefix} Attempting LLM extraction...")
                findings = self._extract_with_llm(imaging_notes)
                evidence["imaging_findings"] = findings
                evidence["metadata"]["extraction_mode"] = "llm"
                extracted_with_llm = True
                logger.info(f"{log_prefix} LLM extraction successful: {len(findings)} findings")
            except Exception as e:
                logger.warning(f"{log_prefix} LLM extraction failed: {e}")

        if not extracted_with_llm:
            logger.info(f"{log_prefix} Falling back to regex extraction...")
            findings = self._extract_with_regex(imaging_notes)
            evidence["imaging_findings"] = findings
            evidence["metadata"]["extraction_mode"] = "regex"
            logger.info(f"{log_prefix} Regex extraction completed: {len(findings)} findings")

        return evidence
