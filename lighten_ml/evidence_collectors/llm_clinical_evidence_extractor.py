"""
LLM-based clinical evidence extractor for myocardial infarction symptoms.

This module provides advanced clinical text extraction using Large Language Models
for better accuracy, context understanding, and flexibility compared to regex patterns.
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


class LLMClinicalEvidenceExtractor(BaseEvidenceCollector):
    """
    Advanced clinical evidence extractor using LLM for structured extraction.

    Provides superior accuracy over regex by understanding:
    - Clinical context and negation
    - Medical abbreviations and synonyms
    - Temporal relationships
    - Severity and quality descriptors
    - Complex medical phrasing
    """

    def __init__(
        self,
        notes_data_loader,
        llm_client: LightenLLMClient,
        max_notes: Optional[int] = None,
    ):
        """
        Initialize the LLM-based clinical evidence extractor.

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
        Collect clinical evidence using LLM-based structured extraction.

        Args:
            patient_id: The ID of the patient
            hadm_id: The ID of the hospital admission

        Returns:
            Dictionary containing structured clinical evidence with confidence scores
        """
        start_time = time.time()
        logger.info(
            f"[LLM-CLINICAL] Starting LLM-based clinical evidence extraction for patient {patient_id}, admission {hadm_id}"
        )
        logger.debug(
            f"[LLM-CLINICAL] Extraction parameters: max_notes={self.max_notes}, llm_available={self.llm_client is not None}"
        )

        evidence = self._get_evidence_base()
        evidence["symptoms"] = []
        evidence["symptom_onset_dates"] = []
        evidence["extraction_method"] = "llm_structured"
        evidence["confidence_scores"] = {}
        evidence["negated_symptoms"] = []

        # Get clinical notes for the admission
        logger.debug(
            f"[LLM-CLINICAL] Loading clinical notes for patient {patient_id}, admission {hadm_id}"
        )
        notes = self.notes_data_loader.get_patient_notes(patient_id, hadm_id)

        if notes.empty:
            logger.warning(
                f"[LLM-CLINICAL] No clinical notes found for patient {patient_id}, admission {hadm_id}"
            )
            evidence["extraction_details"] = {
                "reason": "no_notes_found",
                "processing_time": time.time() - start_time,
            }
            return evidence

        logger.info(
            f"[LLM-CLINICAL] Found {len(notes)} total clinical notes for patient {patient_id}"
        )

        # Filter for relevant note types (discharge summaries, progress notes, etc.)
        relevant_note_types = [
            "Discharge summary",
            "Progress note",
            "Physician note",
            "Nursing note",
            "History and physical",
            "Admission note",
        ]
        relevant_notes = notes[notes["note_type"].isin(relevant_note_types)]

        logger.info(
            f"[LLM-CLINICAL] Filtered to {len(relevant_notes)} relevant notes from {len(notes)} total notes"
        )
        logger.debug(
            f"[LLM-CLINICAL] Note type distribution: {notes['note_type'].value_counts().to_dict()}"
        )

        # Limit number of notes if configured
        original_count = len(relevant_notes)
        if self.max_notes and len(relevant_notes) > self.max_notes:
            relevant_notes = relevant_notes.head(self.max_notes)
            logger.info(
                f"[LLM-CLINICAL] Limited processing from {original_count} to {self.max_notes} notes for efficiency"
            )

        if relevant_notes.empty:
            logger.warning(
                f"[LLM-CLINICAL] No relevant clinical notes found for patient {patient_id}, admission {hadm_id}"
            )
            logger.debug(
                f"[LLM-CLINICAL] Available note types: {list(notes['note_type'].unique())}"
            )
            evidence["extraction_details"] = {
                "reason": "no_relevant_notes",
                "processing_time": time.time() - start_time,
            }
            return evidence

        # Process notes with LLM
        logger.info(
            f"[LLM-CLINICAL] Beginning LLM extraction on {len(relevant_notes)} notes"
        )
        all_symptoms = []
        all_onset_dates = []
        all_negated = []
        confidence_scores = {}
        processing_stats = {
            "successful_notes": 0,
            "failed_notes": 0,
            "total_llm_calls": 0,
            "total_tokens_used": 0,
        }

        for idx, (_, note) in enumerate(relevant_notes.iterrows(), 1):
            note_start_time = time.time()
            logger.debug(
                f"[LLM-CLINICAL] Processing note {idx}/{len(relevant_notes)} from {note['chartdate']} (type: {note.get('note_type', 'unknown')})"
            )

            try:
                note_evidence = self._extract_symptoms_from_note(
                    note["text"], note["chartdate"]
                )
                processing_stats["total_llm_calls"] += 1

                if note_evidence:
                    note_symptoms = note_evidence.get("symptoms", [])
                    note_onset_dates = note_evidence.get("onset_dates", [])
                    note_negated = note_evidence.get("negated_symptoms", [])

                    all_symptoms.extend(note_symptoms)
                    all_onset_dates.extend(note_onset_dates)
                    all_negated.extend(note_negated)

                    logger.info(
                        f"[LLM-CLINICAL] Note {idx} extracted: {len(note_symptoms)} symptoms, {len(note_negated)} negated symptoms"
                    )
                    logger.debug(
                        f"[LLM-CLINICAL] Note {idx} symptoms: {[s.get('name', 'unknown') for s in note_symptoms]}"
                    )

                    # Merge confidence scores
                    for symptom, score in note_evidence.get(
                        "confidence_scores", {}
                    ).items():
                        if (
                            symptom not in confidence_scores
                            or score > confidence_scores[symptom]
                        ):
                            confidence_scores[symptom] = score
                            logger.debug(
                                f"[LLM-CLINICAL] Updated confidence for {symptom}: {score:.3f}"
                            )

                    processing_stats["successful_notes"] += 1
                else:
                    logger.warning(f"[LLM-CLINICAL] Note {idx} returned no evidence")

                note_processing_time = time.time() - note_start_time
                logger.debug(
                    f"[LLM-CLINICAL] Note {idx} processing completed in {note_processing_time:.2f}s"
                )

            except Exception as e:
                processing_stats["failed_notes"] += 1
                logger.error(
                    f"[LLM-CLINICAL] Error processing note {idx} from {note['chartdate']}: {str(e)}"
                )
                logger.debug(
                    f"[LLM-CLINICAL] Failed note text preview: {note['text'][:200]}..."
                )
                continue

        # Deduplicate and structure results
        logger.info(
            f"[LLM-CLINICAL] Deduplicating and structuring results from {len(all_symptoms)} raw symptoms"
        )

        evidence["symptoms"] = self._deduplicate_symptoms(all_symptoms)
        evidence["symptom_onset_dates"] = self._deduplicate_dates(all_onset_dates)
        evidence["negated_symptoms"] = list(set(all_negated))
        evidence["confidence_scores"] = confidence_scores

        # Add detailed extraction statistics
        total_processing_time = time.time() - start_time
        processing_stats["total_processing_time"] = total_processing_time
        processing_stats["avg_time_per_note"] = (
            total_processing_time / len(relevant_notes)
            if relevant_notes.empty == False
            else 0
        )

        evidence["extraction_details"] = {
            "method": "llm_structured",
            "processing_stats": processing_stats,
            "notes_processed": len(relevant_notes),
            "deduplication_stats": {
                "raw_symptoms": len(all_symptoms),
                "final_symptoms": len(evidence["symptoms"]),
                "raw_onset_dates": len(all_onset_dates),
                "final_onset_dates": len(evidence["symptom_onset_dates"]),
                "negated_symptoms": len(evidence["negated_symptoms"]),
            },
        }

        # Calculate and log confidence statistics
        if confidence_scores:
            avg_confidence = sum(confidence_scores.values()) / len(confidence_scores)
            max_confidence = max(confidence_scores.values())
            min_confidence = min(confidence_scores.values())

            logger.info(
                f"[LLM-CLINICAL] Extraction completed in {total_processing_time:.2f}s: "
                f"{len(evidence['symptoms'])} symptoms, {len(evidence['negated_symptoms'])} negated"
            )
            logger.info(
                f"[LLM-CLINICAL] Confidence stats - avg: {avg_confidence:.3f}, "
                f"max: {max_confidence:.3f}, min: {min_confidence:.3f}"
            )
            logger.info(
                f"[LLM-CLINICAL] Processing stats - {processing_stats['successful_notes']}/{len(relevant_notes)} notes successful, "
                f"{processing_stats['total_llm_calls']} LLM calls"
            )
        else:
            logger.warning(
                f"[LLM-CLINICAL] No symptoms extracted with confidence scores"
            )

        # Log high-confidence findings
        high_confidence_symptoms = [
            s for s in evidence["symptoms"] if s.get("confidence", 0) > 0.8
        ]
        if high_confidence_symptoms:
            logger.info(
                f"[LLM-CLINICAL] High-confidence symptoms ({len(high_confidence_symptoms)}): "
                f"{[s.get('name', 'unknown') for s in high_confidence_symptoms]}"
            )

        return evidence

    def _extract_symptoms_from_note(
        self, note_text: str, chart_date: datetime
    ) -> Optional[Dict[str, Any]]:
        """
        Extract structured symptom information from a single clinical note using LLM.

        Args:
            note_text: Text content of the clinical note
            chart_date: Date when the note was written

        Returns:
            Dictionary with extracted symptoms, onset dates, and confidence scores
        """
        if not self.llm_client:
            logger.warning(
                "[LLM-CLINICAL] No LLM client available, skipping LLM extraction"
            )
            return None

        # Log note processing details
        note_length = len(note_text)
        logger.debug(
            f"[LLM-CLINICAL] Processing note of {note_length} characters from {chart_date}"
        )
        logger.debug(f"[LLM-CLINICAL] Note preview: {note_text[:150]}...")

        # Create structured extraction prompt
        prompt_start_time = time.time()
        prompt = self._create_symptom_extraction_prompt(note_text)
        prompt_creation_time = time.time() - prompt_start_time

        logger.debug(
            f"[LLM-CLINICAL] Created extraction prompt in {prompt_creation_time:.3f}s (length: {len(prompt)} chars)"
        )

        try:
            # Get structured response from LLM
            llm_start_time = time.time()
            logger.debug(f"[LLM-CLINICAL] Sending extraction request to LLM...")

            response = self.llm_client.extract_json(prompt)

            llm_processing_time = time.time() - llm_start_time
            logger.debug(
                f"[LLM-CLINICAL] LLM response received in {llm_processing_time:.2f}s"
            )

            if not response:
                logger.warning(
                    f"[LLM-CLINICAL] No response from LLM for symptom extraction (note from {chart_date})"
                )
                return None

            logger.debug(
                f"[LLM-CLINICAL] LLM response keys: {list(response.keys()) if isinstance(response, dict) else 'non-dict response'}"
            )

            # Process and validate the response
            processing_start_time = time.time()
            processed_result = self._process_llm_response(response, chart_date)
            processing_time = time.time() - processing_start_time

            logger.debug(
                f"[LLM-CLINICAL] Response processing completed in {processing_time:.3f}s"
            )

            if processed_result:
                symptoms_count = len(processed_result.get("symptoms", []))
                negated_count = len(processed_result.get("negated_symptoms", []))
                logger.debug(
                    f"[LLM-CLINICAL] Successfully extracted {symptoms_count} symptoms, {negated_count} negated from note"
                )

            return processed_result

        except Exception as e:
            logger.error(
                f"[LLM-CLINICAL] Error in LLM symptom extraction for note from {chart_date}: {str(e)}"
            )
            logger.debug(
                f"[LLM-CLINICAL] Error details - note length: {note_length}, prompt length: {len(prompt)}"
            )
            return None

    def _create_symptom_extraction_prompt(self, note_text: str) -> str:
        """
        Create a structured prompt for symptom extraction.

        Args:
            note_text: Clinical note text

        Returns:
            Formatted prompt for LLM
        """
        prompt = f"""
You are a clinical expert analyzing medical notes for myocardial infarction (MI) symptoms. 
Extract MI-related symptoms with high precision, paying attention to negation and context.

EXTRACT ONLY SYMPTOMS RELATED TO MYOCARDIAL ISCHEMIA:
- Chest pain, pressure, tightness, discomfort, burning
- Substernal pain or pressure
- Pain radiating to arm, jaw, neck, back, shoulder
- Dyspnea (shortness of breath)
- Diaphoresis (sweating)
- Nausea, vomiting
- Lightheadedness, dizziness, syncope
- Palpitations
- Fatigue, weakness (in cardiac context)

IMPORTANT RULES:
1. Do NOT extract symptoms that are explicitly DENIED or NEGATED
2. Focus on CURRENT or RECENT symptoms, not distant history
3. Consider clinical context - symptoms must be potentially MI-related
4. Provide confidence scores (0.0-1.0) based on clarity and clinical relevance
5. Extract onset dates/times when mentioned

Return ONLY valid JSON in this exact format:
{{
    "symptoms": [
        {{
            "name": "chest_pain",
            "description": "crushing chest pain",
            "severity": "severe|moderate|mild|unknown",
            "quality": "crushing|burning|pressure|sharp|unknown",
            "location": "substernal|left chest|etc",
            "radiation": ["left arm", "jaw"],
            "confidence": 0.95
        }}
    ],
    "onset_dates": [
        {{
            "symptom": "chest_pain",
            "onset_time": "2023-01-15 14:30:00",
            "confidence": 0.8
        }}
    ],
    "negated_symptoms": ["chest pain", "dyspnea"],
    "confidence_scores": {{
        "chest_pain": 0.95,
        "dyspnea": 0.0
    }}
}}

Clinical Note:
{note_text[:2000]}
"""
        return prompt

    def _process_llm_response(
        self, response: Dict[str, Any], chart_date: datetime
    ) -> Dict[str, Any]:
        """
        Process and validate LLM response for symptom extraction.

        Args:
            response: Raw LLM response
            chart_date: Date of the clinical note

        Returns:
            Processed and validated symptom data
        """
        logger.debug(
            f"[LLM-CLINICAL] Processing LLM response for note from {chart_date}"
        )

        processed = {
            "symptoms": [],
            "onset_dates": [],
            "negated_symptoms": [],
            "confidence_scores": {},
        }

        # Process symptoms
        raw_symptoms = response.get("symptoms", [])
        logger.debug(
            f"[LLM-CLINICAL] Processing {len(raw_symptoms)} raw symptoms from LLM response"
        )

        confidence_threshold = 0.3
        filtered_symptoms = 0

        for idx, symptom in enumerate(raw_symptoms):
            if isinstance(symptom, dict):
                symptom_confidence = symptom.get("confidence", 0)
                symptom_name = symptom.get("name", "unknown")

                if symptom_confidence > confidence_threshold:
                    processed_symptom = {
                        "name": symptom_name,
                        "description": symptom.get("description", ""),
                        "severity": symptom.get("severity", "unknown"),
                        "quality": symptom.get("quality", "unknown"),
                        "location": symptom.get("location", "unknown"),
                        "radiation": symptom.get("radiation", []),
                        "confidence": symptom_confidence,
                        "chart_date": chart_date.isoformat() if chart_date else None,
                    }
                    processed["symptoms"].append(processed_symptom)
                    logger.debug(
                        f"[LLM-CLINICAL] Accepted symptom {idx+1}: {symptom_name} (confidence: {symptom_confidence:.3f})"
                    )
                else:
                    filtered_symptoms += 1
                    logger.debug(
                        f"[LLM-CLINICAL] Filtered symptom {idx+1}: {symptom_name} (confidence: {symptom_confidence:.3f} < {confidence_threshold})"
                    )
            else:
                logger.warning(
                    f"[LLM-CLINICAL] Invalid symptom format at index {idx}: {type(symptom)}"
                )

        if filtered_symptoms > 0:
            logger.info(
                f"[LLM-CLINICAL] Filtered {filtered_symptoms} low-confidence symptoms (threshold: {confidence_threshold})"
            )

        # Process onset dates
        raw_onset_dates = response.get("onset_dates", [])
        logger.debug(
            f"[LLM-CLINICAL] Processing {len(raw_onset_dates)} raw onset dates from LLM response"
        )

        onset_confidence_threshold = 0.5
        valid_onset_dates = 0
        invalid_onset_dates = 0

        for idx, onset in enumerate(raw_onset_dates):
            if (
                isinstance(onset, dict)
                and onset.get("confidence", 0) > onset_confidence_threshold
            ):
                try:
                    onset_time = pd.to_datetime(onset.get("onset_time"))
                    processed_onset = {
                        "symptom": onset.get("symptom"),
                        "onset_time": onset_time.isoformat(),
                        "confidence": onset.get("confidence", 0.0),
                    }
                    processed["onset_dates"].append(processed_onset)
                    valid_onset_dates += 1
                    logger.debug(
                        f"[LLM-CLINICAL] Accepted onset date {idx+1}: {onset.get('symptom')} at {onset_time} (confidence: {onset.get('confidence', 0):.3f})"
                    )
                except Exception as e:
                    invalid_onset_dates += 1
                    logger.debug(
                        f"[LLM-CLINICAL] Invalid onset date {idx+1}: {onset.get('onset_time')} - {str(e)}"
                    )
                    continue
            else:
                invalid_onset_dates += 1
                logger.debug(
                    f"[LLM-CLINICAL] Filtered onset date {idx+1}: confidence {onset.get('confidence', 0):.3f} < {onset_confidence_threshold}"
                )

        if valid_onset_dates > 0:
            logger.debug(
                f"[LLM-CLINICAL] Processed {valid_onset_dates} valid onset dates, {invalid_onset_dates} filtered/invalid"
            )

        # Process negated symptoms and confidence scores
        raw_negated = response.get("negated_symptoms", [])
        processed["negated_symptoms"] = raw_negated
        processed["confidence_scores"] = response.get("confidence_scores", {})

        logger.debug(
            f"[LLM-CLINICAL] Processed {len(raw_negated)} negated symptoms, {len(processed['confidence_scores'])} confidence scores"
        )

        # Log processing summary
        logger.debug(
            f"[LLM-CLINICAL] Response processing complete: {len(processed['symptoms'])} symptoms, "
            f"{len(processed['onset_dates'])} onset dates, {len(processed['negated_symptoms'])} negated"
        )

        return processed

    def _deduplicate_symptoms(
        self, symptoms: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Remove duplicate symptoms, keeping the highest confidence version.

        Args:
            symptoms: List of symptom dictionaries

        Returns:
            Deduplicated list of symptoms
        """
        symptom_map = {}

        for symptom in symptoms:
            name = symptom.get("name", "unknown")
            confidence = symptom.get("confidence", 0.0)

            if name not in symptom_map or confidence > symptom_map[name].get(
                "confidence", 0.0
            ):
                symptom_map[name] = symptom

        return list(symptom_map.values())

    def _deduplicate_dates(self, dates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate onset dates, keeping the highest confidence version.

        Args:
            dates: List of onset date dictionaries

        Returns:
            Deduplicated list of onset dates
        """
        date_map = {}

        for date_info in dates:
            key = f"{date_info.get('symptom')}_{date_info.get('onset_time')}"
            confidence = date_info.get("confidence", 0.0)

            if key not in date_map or confidence > date_map[key].get("confidence", 0.0):
                date_map[key] = date_info

        return list(date_map.values())
