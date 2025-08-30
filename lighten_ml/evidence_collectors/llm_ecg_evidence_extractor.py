"""
LLM-based ECG evidence extractor for myocardial infarction findings.

This module provides advanced ECG interpretation using Large Language Models
for better accuracy in detecting ischemic changes, Q waves, and other MI-related findings.
"""

import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from ..llm_client import LightenLLMClient
from .base_evidence_collector import BaseEvidenceCollector

logger = logging.getLogger(__name__)


class LLMECGEvidenceExtractor(BaseEvidenceCollector):
    """
    Advanced ECG evidence extractor using LLM for structured interpretation.

    Provides superior accuracy over regex by understanding:
    - Clinical ECG interpretation context
    - Temporal relationships (new vs old changes)
    - Lead-specific findings and anatomical correlations
    - Severity and clinical significance
    - Complex medical terminology and abbreviations
    """

    def __init__(
        self,
        notes_data_loader,
        llm_client: LightenLLMClient,
        max_notes: Optional[int] = None,
    ):
        """
        Initialize the LLM-based ECG evidence extractor.

        Args:
            notes_data_loader: Data loader for clinical notes
            llm_client: LLM client for structured extraction
            max_notes: Maximum number of notes to process per admission
        """
        super().__init__(
            notes_data_loader=notes_data_loader,
            llm_client=llm_client,
            max_notes=max_notes,
        )

    def collect_evidence(self, patient_id: str, hadm_id: str) -> Dict[str, Any]:
        """
        Collect ECG evidence using LLM-based structured extraction.

        Args:
            patient_id: The ID of the patient
            hadm_id: The ID of the hospital admission

        Returns:
            Dictionary containing structured ECG evidence with confidence scores
        """
        start_time = time.time()
        logger.info(
            f"[LLM-ECG] Starting LLM-based ECG evidence extraction for patient {patient_id}, admission {hadm_id}"
        )
        logger.debug(
            f"[LLM-ECG] Extraction parameters: max_notes={self.max_notes}, llm_available={self.llm_client is not None}"
        )

        evidence = self._get_evidence_base()
        evidence["ecg_findings"] = []
        evidence["extraction_method"] = "llm_structured"
        evidence["confidence_scores"] = {}
        evidence["lead_specific_findings"] = {}
        evidence["temporal_context"] = {}

        # Get clinical notes for the admission
        logger.debug(
            f"[LLM-ECG] Loading clinical notes for ECG analysis - patient {patient_id}, admission {hadm_id}"
        )
        notes = self.notes_data_loader.get_patient_notes(patient_id, hadm_id)

        if notes.empty:
            logger.warning(
                f"[LLM-ECG] No clinical notes found for patient {patient_id}, admission {hadm_id}"
            )
            evidence["extraction_details"] = {
                "reason": "no_notes_found",
                "processing_time": time.time() - start_time,
            }
            return evidence

        logger.info(
            f"[LLM-ECG] Found {len(notes)} total clinical notes for ECG analysis"
        )

        # Filter for ECG-related notes and reports
        logger.debug(f"[LLM-ECG] Filtering for ECG-related notes and reports")
        ecg_notes = notes[
            (notes["note_type"] == "ECG")
            | (
                notes["text"].str.contains(
                    "ECG|EKG|electrocardiogram", case=False, na=False
                )
            )
        ]

        logger.info(
            f"[LLM-ECG] Filtered to {len(ecg_notes)} ECG-related notes from {len(notes)} total notes"
        )
        logger.debug(
            f"[LLM-ECG] ECG note type distribution: {ecg_notes['note_type'].value_counts().to_dict() if not ecg_notes.empty else 'no ECG notes'}"
        )

        # Limit number of notes if configured
        original_ecg_count = len(ecg_notes)
        if self.max_notes and len(ecg_notes) > self.max_notes:
            ecg_notes = ecg_notes.head(self.max_notes)
            logger.info(
                f"[LLM-ECG] Limited ECG processing from {original_ecg_count} to {self.max_notes} notes for efficiency"
            )

        if ecg_notes.empty:
            logger.warning(
                f"[LLM-ECG] No ECG notes found for patient {patient_id}, admission {hadm_id}"
            )
            logger.debug(
                f"[LLM-ECG] Available note types: {list(notes['note_type'].unique())}"
            )
            evidence["extraction_details"] = {
                "reason": "no_ecg_notes",
                "processing_time": time.time() - start_time,
            }
            return evidence

        # Process ECG notes with LLM
        logger.info(
            f"[LLM-ECG] Beginning LLM ECG interpretation on {len(ecg_notes)} notes"
        )
        all_findings = []
        confidence_scores = {}
        lead_findings = {}
        temporal_context = {}
        processing_stats = {
            "successful_notes": 0,
            "failed_notes": 0,
            "total_llm_calls": 0,
            "mi_related_findings": 0,
        }

        for idx, (_, note) in enumerate(ecg_notes.iterrows(), 1):
            note_start_time = time.time()
            logger.debug(
                f"[LLM-ECG] Processing ECG note {idx}/{len(ecg_notes)} from {note['chartdate']} (type: {note.get('note_type', 'unknown')})"
            )

            try:
                note_evidence = self._extract_ecg_from_note(
                    note["text"], note["chartdate"]
                )
                processing_stats["total_llm_calls"] += 1

                if note_evidence:
                    note_findings = note_evidence.get("ecg_findings", [])
                    mi_related_count = sum(
                        1 for f in note_findings if f.get("mi_related", False)
                    )

                    all_findings.extend(note_findings)
                    processing_stats["mi_related_findings"] += mi_related_count

                    logger.info(
                        f"[LLM-ECG] Note {idx} extracted: {len(note_findings)} findings ({mi_related_count} MI-related)"
                    )
                    logger.debug(
                        f"[LLM-ECG] Note {idx} findings: {[f.get('finding', 'unknown') for f in note_findings]}"
                    )

                    # Merge confidence scores
                    for finding, score in note_evidence.get(
                        "confidence_scores", {}
                    ).items():
                        if (
                            finding not in confidence_scores
                            or score > confidence_scores[finding]
                        ):
                            confidence_scores[finding] = score
                            logger.debug(
                                f"[LLM-ECG] Updated confidence for {finding}: {score:.3f}"
                            )

                    # Merge lead-specific findings
                    note_lead_findings = note_evidence.get("lead_specific_findings", {})
                    for lead, findings in note_lead_findings.items():
                        if lead not in lead_findings:
                            lead_findings[lead] = []
                        lead_findings[lead].extend(findings)
                        logger.debug(
                            f"[LLM-ECG] Added {len(findings)} findings for lead {lead}"
                        )

                    temporal_context.update(note_evidence.get("temporal_context", {}))
                    processing_stats["successful_notes"] += 1
                else:
                    logger.warning(f"[LLM-ECG] Note {idx} returned no ECG evidence")

                note_processing_time = time.time() - note_start_time
                logger.debug(
                    f"[LLM-ECG] Note {idx} processing completed in {note_processing_time:.2f}s"
                )

            except Exception as e:
                processing_stats["failed_notes"] += 1
                logger.error(
                    f"[LLM-ECG] Error processing ECG note {idx} from {note['chartdate']}: {str(e)}"
                )
                logger.debug(
                    f"[LLM-ECG] Failed note text preview: {note['text'][:200]}..."
                )
                continue

        # Deduplicate and structure results
        logger.info(
            f"[LLM-ECG] Deduplicating and structuring results from {len(all_findings)} raw ECG findings"
        )

        evidence["ecg_findings"] = self._deduplicate_findings(all_findings)
        evidence["confidence_scores"] = confidence_scores
        evidence["lead_specific_findings"] = lead_findings
        evidence["temporal_context"] = temporal_context

        # Add detailed extraction statistics
        total_processing_time = time.time() - start_time
        processing_stats["total_processing_time"] = total_processing_time
        processing_stats["avg_time_per_note"] = (
            total_processing_time / len(ecg_notes) if not ecg_notes.empty else 0
        )

        evidence["extraction_details"] = {
            "method": "llm_structured_ecg",
            "processing_stats": processing_stats,
            "notes_processed": len(ecg_notes),
            "deduplication_stats": {
                "raw_findings": len(all_findings),
                "final_findings": len(evidence["ecg_findings"]),
                "leads_analyzed": len(lead_findings),
                "mi_related_findings": processing_stats["mi_related_findings"],
            },
        }

        # Calculate and log confidence statistics
        if confidence_scores:
            avg_confidence = sum(confidence_scores.values()) / len(confidence_scores)
            max_confidence = max(confidence_scores.values())
            min_confidence = min(confidence_scores.values())

            logger.info(
                f"[LLM-ECG] ECG extraction completed in {total_processing_time:.2f}s: "
                f"{len(evidence['ecg_findings'])} findings ({processing_stats['mi_related_findings']} MI-related)"
            )
            logger.info(
                f"[LLM-ECG] Confidence stats - avg: {avg_confidence:.3f}, "
                f"max: {max_confidence:.3f}, min: {min_confidence:.3f}"
            )
            logger.info(
                f"[LLM-ECG] Processing stats - {processing_stats['successful_notes']}/{len(ecg_notes)} notes successful, "
                f"{processing_stats['total_llm_calls']} LLM calls"
            )
        else:
            logger.warning(
                f"[LLM-ECG] No ECG findings extracted with confidence scores"
            )

        # Log high-confidence MI-related findings
        mi_findings = [
            f
            for f in evidence["ecg_findings"]
            if f.get("mi_related", False) and f.get("confidence", 0) > 0.7
        ]
        if mi_findings:
            logger.info(
                f"[LLM-ECG] High-confidence MI-related findings ({len(mi_findings)}): "
                f"{[f.get('finding', 'unknown') for f in mi_findings]}"
            )

        # Log anatomical regions affected
        regions = set(
            f.get("anatomical_region", "unknown")
            for f in mi_findings
            if f.get("anatomical_region") != "unknown"
        )
        if regions:
            logger.info(
                f"[LLM-ECG] Anatomical regions with MI-related changes: {list(regions)}"
            )

        return evidence

    def _extract_ecg_from_note(
        self, note_text: str, chart_date: datetime
    ) -> Optional[Dict[str, Any]]:
        """
        Extract structured ECG information from a single note using LLM.

        Args:
            note_text: Text content of the ECG note/report
            chart_date: Date when the ECG was performed

        Returns:
            Dictionary with extracted ECG findings and confidence scores
        """
        if not self.llm_client:
            logger.warning("No LLM client available, skipping LLM extraction")
            return None

        # Create structured extraction prompt
        prompt = self._create_ecg_extraction_prompt(note_text)

        try:
            # Get structured response from LLM
            response = self.llm_client.extract_json(prompt)

            if not response:
                logger.warning("No response from LLM for ECG extraction")
                return None

            # Process and validate the response
            return self._process_ecg_llm_response(response, chart_date)

        except Exception as e:
            logger.error(f"Error in LLM ECG extraction: {str(e)}")
            return None

    def _create_ecg_extraction_prompt(self, note_text: str) -> str:
        """
        Create a structured prompt for ECG finding extraction.

        Args:
            note_text: ECG note/report text

        Returns:
            Formatted prompt for LLM
        """
        prompt = f"""
You are a cardiology expert analyzing ECG reports for myocardial infarction (MI) findings.
Extract ONLY MI-related ECG changes with high clinical precision.

EXTRACT THESE MI-RELATED ECG FINDINGS (Criteria B.2 and B.3):

1. ST ELEVATION (≥1mm in ≥2 contiguous leads):
   - New ST elevation, STEMI
   - Specify leads and magnitude if mentioned

2. ST DEPRESSION (≥0.5mm horizontal/down-sloping in ≥2 contiguous leads):
   - New ST depression
   - Specify leads and magnitude if mentioned

3. T WAVE INVERSION (≥1mm in ≥2 contiguous leads with prominent R wave):
   - New T wave inversion
   - Specify leads affected

4. PATHOLOGICAL Q WAVES (≥0.02 seconds or QS complex in ≥2 contiguous leads):
   - New Q waves, pathological Q waves
   - QS complexes
   - Specify leads affected

5. OTHER MI-RELATED FINDINGS:
   - New LBBB (left bundle branch block)
   - Ventricular arrhythmias (VT/VF)

CRITICAL RULES:
1. Focus on NEW or ACUTE changes (not old/chronic findings)
2. Ignore non-ischemic findings (atrial fib, sinus tach, etc.)
3. Pay attention to lead groups and contiguous leads
4. Consider clinical context and physician interpretation
5. Provide confidence based on clarity and clinical significance

Return ONLY valid JSON in this exact format:
{{
    "ecg_findings": [
        {{
            "finding": "ST_elevation",
            "description": "ST elevation in leads II, III, aVF",
            "leads": ["II", "III", "aVF"],
            "magnitude": "2-3mm",
            "anatomical_region": "inferior",
            "is_new": true,
            "mi_related": true,
            "confidence": 0.95,
            "clinical_significance": "high"
        }}
    ],
    "lead_specific_findings": {{
        "II": ["ST elevation 2mm"],
        "III": ["ST elevation 3mm"],
        "aVF": ["ST elevation 2mm"]
    }},
    "temporal_context": {{
        "new_changes": true,
        "comparison_available": false,
        "acute_findings": true
    }},
    "confidence_scores": {{
        "ST_elevation": 0.95,
        "pathological_Q_waves": 0.0
    }}
}}

ECG Report:
{note_text[:2000]}
"""
        return prompt

    def _process_ecg_llm_response(
        self, response: Dict[str, Any], chart_date: datetime
    ) -> Dict[str, Any]:
        """
        Process and validate LLM response for ECG extraction.

        Args:
            response: Raw LLM response
            chart_date: Date of the ECG

        Returns:
            Processed and validated ECG data
        """
        processed = {
            "ecg_findings": [],
            "lead_specific_findings": {},
            "temporal_context": {},
            "confidence_scores": {},
        }

        # Process ECG findings
        for finding in response.get("ecg_findings", []):
            if (
                isinstance(finding, dict) and finding.get("confidence", 0) > 0.4
            ):  # Confidence threshold
                processed_finding = {
                    "finding": finding.get("finding", "unknown"),
                    "description": finding.get("description", ""),
                    "leads": finding.get("leads", []),
                    "magnitude": finding.get("magnitude", "unknown"),
                    "anatomical_region": finding.get("anatomical_region", "unknown"),
                    "is_new": finding.get("is_new", False),
                    "mi_related": finding.get("mi_related", False),
                    "confidence": finding.get("confidence", 0.0),
                    "clinical_significance": finding.get(
                        "clinical_significance", "unknown"
                    ),
                    "chart_date": chart_date.isoformat() if chart_date else None,
                }
                processed["ecg_findings"].append(processed_finding)

        # Process lead-specific findings
        processed["lead_specific_findings"] = response.get("lead_specific_findings", {})

        # Process temporal context
        processed["temporal_context"] = response.get("temporal_context", {})

        # Process confidence scores
        processed["confidence_scores"] = response.get("confidence_scores", {})

        return processed

    def _deduplicate_findings(
        self, findings: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Remove duplicate ECG findings, keeping the highest confidence version.

        Args:
            findings: List of ECG finding dictionaries

        Returns:
            Deduplicated list of findings
        """
        finding_map = {}

        for finding in findings:
            key = f"{finding.get('finding')}_{finding.get('anatomical_region')}"
            confidence = finding.get("confidence", 0.0)

            if key not in finding_map or confidence > finding_map[key].get(
                "confidence", 0.0
            ):
                finding_map[key] = finding

        return list(finding_map.values())

    def get_mi_related_findings(self, evidence: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Filter ECG findings to return only MI-related findings.

        Args:
            evidence: ECG evidence dictionary

        Returns:
            List of MI-related ECG findings
        """
        mi_findings = []

        for finding in evidence.get("ecg_findings", []):
            if finding.get("mi_related", False) and finding.get("confidence", 0) > 0.5:
                mi_findings.append(finding)

        return mi_findings

    def get_anatomical_regions_affected(self, evidence: Dict[str, Any]) -> List[str]:
        """
        Get list of anatomical regions with MI-related ECG changes.

        Args:
            evidence: ECG evidence dictionary

        Returns:
            List of affected anatomical regions
        """
        regions = set()

        for finding in self.get_mi_related_findings(evidence):
            region = finding.get("anatomical_region")
            if region and region != "unknown":
                regions.add(region)

        return list(regions)
