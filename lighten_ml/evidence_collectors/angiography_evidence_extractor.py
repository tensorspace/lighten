"""Evidence extractor for angiography findings related to MI."""

import logging
import re
from typing import Any, Dict, List, Optional

from .base_evidence_collector import BaseEvidenceCollector

logger = logging.getLogger(__name__)


class AngiographyEvidenceExtractor(BaseEvidenceCollector):
    """Extracts angiography evidence (e.g., from cardiac cath reports) for MI."""

    ANGIO_PATTERNS = [
        {
            "name": "Intracoronary Thrombus",
            "pattern": re.compile(r"(thrombus|thrombotic|occlusion)", re.IGNORECASE),
            "mi_related": True,
        }
    ]

    def __init__(
        self,
        notes_data_loader: Any,
        llm_client: Optional[Any] = None,
        max_notes: Optional[int] = None,
    ):
        """Initialize the angiography evidence extractor.

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
        """Collect angiography evidence from notes for a specific admission."""
        log_prefix = f"[{patient_id}][{hadm_id}] [ANGIO_EXTRACTOR]"

        evidence = self._get_evidence_base()
        evidence["angiography_findings"] = []
        evidence["metadata"] = {"extraction_mode": "none"}

        logger.info(f"{log_prefix} Fetching clinical notes for angiography evidence.")
        notes = self.notes_data_loader.get_patient_notes(patient_id, hadm_id)
        # Filter for relevant notes like cardiac cath reports
        cath_notes = notes[notes["note_type"].isin(["Cardiac Cath"])]
        logger.info(f"{log_prefix} Found {len(cath_notes)} cardiac cath reports out of {len(notes)} total notes.")

        # Limit number of notes if configured
        if self.max_notes and len(cath_notes) > self.max_notes:
            logger.info(f"{log_prefix} Limiting notes from {len(cath_notes)} to {self.max_notes}.")
            cath_notes = cath_notes.head(self.max_notes)

        if cath_notes.empty:
            logger.warning(f"{log_prefix} No cardiac cath reports found.")
            return evidence

        logger.info(f"{log_prefix} Processing {len(cath_notes)} cardiac cath reports.")

        try_llm = self.llm_client and self.llm_client.enabled
        logger.info(f"{log_prefix} Method selection: LLM available={try_llm}")

        extracted_with_llm = False
        if try_llm:
            try:
                logger.info(f"{log_prefix} Attempting LLM extraction...")
                findings = self._extract_with_llm(cath_notes)
                evidence["angiography_findings"] = findings
                evidence["metadata"]["extraction_mode"] = "llm"
                extracted_with_llm = True
                logger.info(
                    f"{log_prefix} LLM extraction successful: {len(findings)} findings"
                )
            except Exception as e:
                logger.warning(f"{log_prefix} LLM extraction failed: {e}")

        if not extracted_with_llm:
            logger.info(f"{log_prefix} Falling back to regex extraction...")
            findings = self._extract_with_regex(cath_notes)
            evidence["angiography_findings"] = findings
            evidence["metadata"]["extraction_mode"] = "regex"
            logger.info(f"{log_prefix} Regex extraction completed: {len(findings)} findings")

        return evidence
