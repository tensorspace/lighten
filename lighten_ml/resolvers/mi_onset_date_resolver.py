"""MI Onset Date Resolver using LLM-based clinical documentation analysis."""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..llm_client import LightenLLMClient

logger = logging.getLogger(__name__)


class MIOnsetDateResolver:
    """Resolves MI onset dates from clinical documentation using LLM analysis.

    Follows the clinical guideline hierarchy:
    1. Symptom Onset Date (HIGHEST PRIORITY)
    2. First Abnormal ECG Date
    3. First Elevated Troponin Date
    4. Clinical Recognition/Diagnosis Date
    5. Hospital Presentation Date
    """

    def __init__(self):
        """Initialize the MI Onset Date Resolver with LLM client."""
        self.llm_client = LightenLLMClient()

        # Clinical guideline hierarchy priorities
        self.DATE_HIERARCHY = {
            "symptom_onset": 1,
            "first_abnormal_ecg": 2,
            "first_elevated_troponin": 3,
            "clinical_recognition": 4,
            "hospital_presentation": 5,
        }

    def extract_mi_onset_date(
        self,
        clinical_notes: str,
        troponin_data: List[Dict],
        admission_date: str,
        patient_id: str,
        hadm_id: str,
    ) -> Dict[str, Any]:
        """Extract MI onset date using LLM analysis of clinical documentation.

        Args:
            clinical_notes: Raw clinical documentation text
            troponin_data: List of troponin test results with timestamps
            admission_date: Hospital admission date
            patient_id: Patient identifier
            hadm_id: Hospital admission identifier

        Returns:
            Dictionary containing onset date analysis results
        """
        logger.info(
            f"[{hadm_id}] [EXTRACT] MI ONSET DATE EXTRACTION - Starting LLM analysis"
        )

        try:
            # Use LLM to extract dates according to clinical guideline hierarchy
            llm_result = self._llm_extract_onset_dates(
                clinical_notes, troponin_data, admission_date
            )

            # Process and validate the LLM results
            processed_result = self._process_llm_results(
                llm_result, troponin_data, admission_date, hadm_id
            )

            # Apply clinical guideline hierarchy to select final date
            final_result = self._apply_guideline_hierarchy(processed_result, hadm_id)

            logger.info(f"[{hadm_id}] [COMPLETE] MI ONSET DATE EXTRACTION COMPLETE")
            logger.info(
                f"[{hadm_id}] [DATE] Final MI Onset Date: {final_result.get('onset_date', 'Not determined')}"
            )
            logger.info(
                f"[{hadm_id}] [BASIS] Selection Basis: {final_result.get('selection_basis', 'N/A')}"
            )

            return final_result

        except Exception as e:
            logger.error(f"[{hadm_id}] [ERROR] MI ONSET DATE EXTRACTION FAILED: {e}")
            return {
                "onset_date": None,
                "selection_basis": "extraction_failed",
                "confidence": 0.0,
                "error": str(e),
                "hierarchy_analysis": {},
            }

    def _llm_extract_onset_dates(
        self, clinical_notes: str, troponin_data: List[Dict], admission_date: str
    ) -> Dict[str, Any]:
        """Use LLM to extract dates according to clinical guideline hierarchy."""

        # Prepare troponin data summary for LLM context
        troponin_summary = self._prepare_troponin_summary(troponin_data)

        prompt = f"""
You are a clinical expert analyzing medical documentation to determine the MI onset date according to the 4th Universal Definition of Myocardial Infarction guidelines.

**CLINICAL GUIDELINE HIERARCHY (prioritize in order 1-5):**

1. **Symptom Onset Date (HIGHEST PRIORITY)**
   - Date when patient first experienced acute ischemic symptoms
   - Examples: chest pain, substernal pressure, anginal equivalents
   - Look for phrases like "began at", "started", "awoke with", "complained of starting"

2. **First Abnormal ECG Date**
   - Date of first ECG showing acute ischemic changes
   - ST elevation, new ST depression, new T wave inversions, new Q waves
   - Look for "ECG on [date] shows", "first ECG with changes"

3. **First Elevated Troponin Date**
   - Date of first troponin >0.014 ng/mL (diagnostic threshold)
   - Use structured lab data if clinical notes don't specify

4. **Clinical Recognition/Diagnosis Date**
   - Date when clinical team suspected/diagnosed MI
   - "STEMI protocol activated", "diagnosed with MI", "ACS suspected"

5. **Hospital Presentation Date**
   - Only if no other dates available and MI clearly present on arrival
   - "presented with ongoing MI", "admitted in acute phase"

**SPECIAL CONSIDERATIONS:**
- If symptom onset >24 hours before presentation: use symptom onset date
- If symptoms "ongoing" or "persistent": use date symptoms first began
- If intermittent symptoms over days: use final/continuous episode date
- Do not estimate dates - use only documented information

**CLINICAL DOCUMENTATION:**
{clinical_notes}

**TROPONIN DATA:**
{troponin_summary}

**ADMISSION DATE:** {admission_date}

**ANALYSIS REQUIRED:**
Extract and analyze dates for each hierarchy level. For each level found, provide:
1. The specific date found (MM/DD/YYYY format)
2. The exact text evidence supporting this date
3. Confidence level (0.0-1.0)
4. Any relevant time information

**OUTPUT FORMAT (JSON):**
{{
    "symptom_onset": {{
        "date": "MM/DD/YYYY or null",
        "evidence": "exact text from notes",
        "confidence": 0.0-1.0,
        "time_info": "additional time details if available"
    }},
    "first_abnormal_ecg": {{
        "date": "MM/DD/YYYY or null",
        "evidence": "exact text from notes",
        "confidence": 0.0-1.0,
        "time_info": "additional time details if available"
    }},
    "first_elevated_troponin": {{
        "date": "MM/DD/YYYY or null",
        "evidence": "from notes or lab data",
        "confidence": 0.0-1.0,
        "time_info": "additional time details if available"
    }},
    "clinical_recognition": {{
        "date": "MM/DD/YYYY or null",
        "evidence": "exact text from notes",
        "confidence": 0.0-1.0,
        "time_info": "additional time details if available"
    }},
    "hospital_presentation": {{
        "date": "MM/DD/YYYY or null",
        "evidence": "exact text from notes",
        "confidence": 0.0-1.0,
        "time_info": "additional time details if available"
    }},
    "analysis_notes": "any additional clinical reasoning or considerations"
}}

Analyze the documentation carefully and extract dates according to the clinical guideline hierarchy. Be precise and conservative - only extract dates that are clearly documented.
"""

        logger.info("[LLM] MI ONSET DATE EXTRACTION - Sending prompt to LLM...")

        try:
            response = self.llm_client.generate_response(prompt)
            logger.info("[SUCCESS] LLM MI ONSET DATE EXTRACTION - Response received")

            # Parse JSON response
            result = json.loads(response)
            logger.info(
                "[SUCCESS] LLM MI ONSET DATE EXTRACTION - JSON parsed successfully"
            )

            return result

        except json.JSONDecodeError as e:
            logger.error(
                f"[ERROR] LLM MI ONSET DATE EXTRACTION - JSON parsing failed: {e}"
            )
            logger.error(f"Raw LLM response: {response}")
            raise
        except Exception as e:
            logger.error(f"[ERROR] LLM MI ONSET DATE EXTRACTION - LLM call failed: {e}")
            raise

    def _prepare_troponin_summary(self, troponin_data: List[Dict]) -> str:
        """Prepare troponin data summary for LLM context."""
        if not troponin_data:
            return "No troponin data available"

        summary_lines = ["Troponin Test Results:"]
        for i, test in enumerate(troponin_data[:10]):  # Limit to first 10 tests
            value = test.get("value", "N/A")
            unit = test.get("unit", "N/A")
            timestamp = test.get("timestamp", "N/A")
            above_threshold = test.get("above_threshold", False)

            threshold_status = (
                "ABOVE THRESHOLD" if above_threshold else "below threshold"
            )
            summary_lines.append(
                f"  Test {i+1}: {value} {unit} on {timestamp} ({threshold_status})"
            )

        return "\n".join(summary_lines)

    def _process_llm_results(
        self,
        llm_result: Dict[str, Any],
        troponin_data: List[Dict],
        admission_date: str,
        hadm_id: str,
    ) -> Dict[str, Any]:
        """Process and validate LLM extraction results."""

        logger.info(
            f"[{hadm_id}] [PROCESS] PROCESSING LLM RESULTS - Validating extracted dates"
        )

        processed = {"hierarchy_analysis": {}, "validation_notes": []}

        # Process each hierarchy level
        for level_name, priority in self.DATE_HIERARCHY.items():
            level_data = llm_result.get(level_name, {})

            if level_data and level_data.get("date"):
                # Validate date format and parse
                validated_date = self._validate_and_parse_date(
                    level_data.get("date"), hadm_id, level_name
                )

                if validated_date:
                    processed["hierarchy_analysis"][level_name] = {
                        "date": validated_date,
                        "priority": priority,
                        "evidence": level_data.get("evidence", ""),
                        "confidence": level_data.get("confidence", 0.0),
                        "time_info": level_data.get("time_info", ""),
                        "validated": True,
                    }
                    logger.info(
                        f"[{hadm_id}] [VALID] {level_name.upper()}: {validated_date} (confidence: {level_data.get('confidence', 0.0)})"
                    )
                else:
                    processed["validation_notes"].append(
                        f"Invalid date format for {level_name}: {level_data.get('date')}"
                    )
                    logger.warning(
                        f"[{hadm_id}] [WARNING] {level_name.upper()}: Invalid date format - {level_data.get('date')}"
                    )
            else:
                logger.info(f"[{hadm_id}] [NONE] {level_name.upper()}: No date found")

        # Add analysis notes from LLM
        if "analysis_notes" in llm_result:
            processed["llm_analysis_notes"] = llm_result["analysis_notes"]

        return processed

    def _validate_and_parse_date(
        self, date_str: str, hadm_id: str, level_name: str
    ) -> Optional[str]:
        """Validate and parse date string to MM/DD/YYYY format."""
        if not date_str or date_str.lower() == "null":
            return None

        try:
            # Try to parse various date formats
            parsed_date = datetime.strptime(date_str, "%m/%d/%Y")
            return parsed_date.strftime("%m/%d/%Y")
        except ValueError:
            try:
                # Try alternative formats
                for fmt in ["%Y-%m-%d", "%m-%d-%Y", "%m/%d/%y", "%d/%m/%Y"]:
                    try:
                        parsed_date = datetime.strptime(date_str, fmt)
                        return parsed_date.strftime("%m/%d/%Y")
                    except ValueError:
                        continue
            except:
                pass

        logger.warning(
            f"[{hadm_id}] [WARNING] DATE VALIDATION FAILED - {level_name}: {date_str}"
        )
        return None

    def _apply_guideline_hierarchy(
        self, processed_result: Dict[str, Any], hadm_id: str
    ) -> Dict[str, Any]:
        """Apply clinical guideline hierarchy to select the final MI onset date."""

        logger.info(f"[{hadm_id}] [HIERARCHY] APPLYING CLINICAL GUIDELINE HIERARCHY")

        hierarchy_analysis = processed_result.get("hierarchy_analysis", {})

        if not hierarchy_analysis:
            logger.warning(
                f"[{hadm_id}] [WARNING] No valid dates found in hierarchy analysis"
            )
            return {
                "onset_date": None,
                "selection_basis": "no_valid_dates",
                "confidence": 0.0,
                "hierarchy_analysis": {},
                "validation_notes": processed_result.get("validation_notes", []),
            }

        # Sort by priority (1 = highest priority)
        sorted_levels = sorted(
            hierarchy_analysis.items(), key=lambda x: x[1]["priority"]
        )

        # Select the highest priority date with sufficient confidence
        selected_level = None
        selected_data = None

        for level_name, level_data in sorted_levels:
            confidence = level_data.get("confidence", 0.0)
            if confidence >= 0.5:  # Minimum confidence threshold
                selected_level = level_name
                selected_data = level_data
                break

        if selected_level and selected_data:
            logger.info(
                f"[{hadm_id}] [SELECTED] SELECTED DATE: {selected_data['date']} from {selected_level}"
            )
            logger.info(
                f"[{hadm_id}] [EVIDENCE] EVIDENCE: {selected_data.get('evidence', 'N/A')}"
            )

            return {
                "onset_date": selected_data["date"],
                "selection_basis": selected_level,
                "confidence": selected_data["confidence"],
                "evidence": selected_data.get("evidence", ""),
                "time_info": selected_data.get("time_info", ""),
                "hierarchy_analysis": hierarchy_analysis,
                "validation_notes": processed_result.get("validation_notes", []),
                "llm_analysis_notes": processed_result.get("llm_analysis_notes", ""),
            }
        else:
            logger.warning(
                f"[{hadm_id}] [WARNING] No dates met minimum confidence threshold (0.5)"
            )
            return {
                "onset_date": None,
                "selection_basis": "insufficient_confidence",
                "confidence": 0.0,
                "hierarchy_analysis": hierarchy_analysis,
                "validation_notes": processed_result.get("validation_notes", []),
            }
