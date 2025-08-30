"""Evidence extractor for angiography findings related to MI."""

import logging
import re
from typing import Any, Dict, List

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

    def collect_evidence(
        self, patient_id: str, hadm_id: str, text: str
    ) -> Dict[str, Any]:
        """Collect angiography evidence from a given clinical note text."""
        log_prefix = f"[{patient_id}][{hadm_id}] [ANGIO]"
        logger.debug(f"{log_prefix} Starting angiography evidence extraction.")

        evidence = {"angiography_findings": []}
        findings = []

        for definition in self.ANGIO_PATTERNS:
            for match in definition["pattern"].finditer(text):
                finding = {
                    "finding_type": "angiography",
                    "finding_name": definition["name"],
                    "evidence": match.group(0),
                    "mi_related": definition["mi_related"],
                    "source": "regex",
                }
                findings.append(finding)

        if findings:
            logger.info(
                f"{log_prefix} Found {len(findings)} angiography-related findings."
            )
            logger.debug(f"{log_prefix} Details: {findings}")

        evidence["angiography_findings"] = findings
        return evidence
