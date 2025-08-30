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
        evidence = self._get_evidence_base()
        evidence["angiography_findings"] = []
        evidence["metadata"] = {"extraction_mode": "none"}

        notes = self.notes_data_loader.get_patient_notes(patient_id, hadm_id)
        # Filter for relevant notes like cardiac cath reports
        cath_notes = notes[notes["note_type"].isin(["Cardiac Cath"])]

        # Limit number of notes if configured
        if self.max_notes and len(cath_notes) > self.max_notes:
            cath_notes = cath_notes.head(self.max_notes)

        if cath_notes.empty:
            return evidence

        logger.info(
            f"[{hadm_id}] ANGIOGRAPHY EXTRACTION - Method selection: LLM available={self.llm_client and self.llm_client.enabled}"
        )

        extracted_with_llm = False
        if self.llm_client and self.llm_client.enabled:
            try:
                logger.info(
                    f"[{hadm_id}] ANGIOGRAPHY EXTRACTION - Attempting LLM extraction..."
                )
                findings = self._extract_with_llm(cath_notes)
                evidence["angiography_findings"] = findings
                evidence["metadata"]["extraction_mode"] = "llm"
                extracted_with_llm = True
                logger.info(
                    f"[{hadm_id}] ANGIOGRAPHY EXTRACTION - LLM extraction successful: {len(findings)} findings"
                )
            except Exception as e:
                logger.warning(
                    f"[{hadm_id}] ANGIOGRAPHY EXTRACTION - LLM extraction failed: {str(e)}, falling back to regex"
                )

        if not extracted_with_llm:
            evidence["metadata"]["extraction_mode"] = (
                "regex_fallback" if self.llm_client else "regex"
            )
            logger.info(f"[{hadm_id}] ANGIOGRAPHY EXTRACTION - Using regex extraction")
            findings = self._extract_with_regex(cath_notes)
            evidence["angiography_findings"] = findings
            logger.info(
                f"[{hadm_id}] ANGIOGRAPHY EXTRACTION - Regex extraction complete: {len(findings)} findings"
            )

        # Log detailed angiography evidence found
        thrombus_findings = [
            f for f in findings if "thrombus" in f.get("finding", "").lower()
        ]
        if findings:
            logger.info(f"[{hadm_id}] ANGIOGRAPHY EVIDENCE FOUND:")
            for i, finding in enumerate(findings[:3], 1):  # Log first 3 findings
                logger.info(f"[{hadm_id}]   {i}. {finding.get('finding', 'unknown')}")
                logger.info(f"[{hadm_id}]      Vessel: {finding.get('vessel', 'N/A')}")
                logger.info(
                    f"[{hadm_id}]      MI-Related: {finding.get('mi_related', 'N/A')}"
                )
                logger.info(
                    f"[{hadm_id}]      Context: {finding.get('context', 'N/A')[:100]}..."
                )
            if thrombus_findings:
                logger.info(
                    f"[{hadm_id}] ANGIOGRAPHY - Thrombus findings detected: {len(thrombus_findings)}"
                )
        else:
            logger.info(
                f"[{hadm_id}] ANGIOGRAPHY EVIDENCE - No angiography findings detected"
            )

        # Add a summary flag for the rule engine
        evidence["thrombus_present"] = any(
            f["mi_related"] for f in evidence["angiography_findings"]
        )

        return evidence

    def _extract_with_llm(self, notes: Any) -> List[Dict[str, Any]]:
        """Use LLM to extract angiography findings."""
        instructions = (
            "Read the cardiac catheterization report and identify findings of intracoronary thrombus. "
            "Return a JSON object with a key 'angiography_findings', a list of objects. "
            "Each object should have 'finding' ('Intracoronary Thrombus'), 'context', and 'vessel' (e.g., 'LAD', 'RCA')."
        )
        findings = []
        for _, note in notes.iterrows():
            text = str(note.get("text", ""))
            if not text.strip():
                continue
            data = self.llm_client.extract_json(instructions, text)
            for f in data.get("angiography_findings", []) or []:
                findings.append(
                    {
                        "finding": f.get("finding"),
                        "context": f.get("context"),
                        "vessel": f.get("vessel"),
                        "mi_related": True,
                        "charttime": note.get("charttime"),
                    }
                )
        return findings

    def _extract_with_regex(self, notes: Any) -> List[Dict[str, Any]]:
        """Use regex to extract angiography findings."""
        findings = []
        for _, note in notes.iterrows():
            text = note.get("text", "")
            for pattern_info in self.ANGIO_PATTERNS:
                for match in pattern_info["pattern"].finditer(text):
                    context = text[
                        max(0, match.start() - 150) : min(len(text), match.end() + 150)
                    ]
                    findings.append(
                        {
                            "finding": pattern_info["name"],
                            "context": context.strip(),
                            "mi_related": True,
                            "charttime": note.get("charttime"),
                        }
                    )
                    break  # Move to next note after first finding
        return findings
