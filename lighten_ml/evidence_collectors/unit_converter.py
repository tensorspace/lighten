"""Unit conversion utilities for medical lab values using LLM-based conversion."""

import json
import logging
import pandas as pd
from typing import Dict, Optional, Tuple

from ..llm_client import LightenLLMClient

logger = logging.getLogger(__name__)


def convert_troponin_units(value: float, unit: str) -> tuple[float, str]:
    """Convert troponin values to standard ng/mL units using LLM-based conversion.

    Args:
        value: The numeric value
        unit: The original unit

    Returns:
        Tuple of (converted_value, final_unit)
    """
    logger.info(f"[CONVERT] UNIT CONVERSION START: {value} {unit}")

    if not unit or pd.isna(unit):
        logger.warning(
            f"[WARNING] UNIT CONVERSION: No unit provided for troponin value {value}, assuming ng/mL"
        )
        logger.info(
            f"[SUCCESS] UNIT CONVERSION RESULT: {value} (no unit) -> {value} ng/mL (assumed)"
        )
        return value, "ng/mL"

    unit_str = str(unit).strip()
    logger.info(f"[DATA] UNIT CONVERSION: Processing {value} {unit_str}")

    # Quick check for already standard units
    if unit_str.lower() in ["ng/ml", "ng/dl", "nanogram/ml", "nanogram/dl"]:
        logger.info(
            f"[SUCCESS] UNIT CONVERSION: Already in standard format - no conversion needed"
        )
        logger.info(
            f"[SUCCESS] UNIT CONVERSION RESULT: {value} {unit_str} -> {value} ng/mL (no change)"
        )
        return value, "ng/mL"

    # Use LLM for intelligent unit conversion
    logger.info(f"[LLM] UNIT CONVERSION: Attempting LLM-based conversion...")
    try:
        converted_value, final_unit = _llm_convert_troponin_units(value, unit_str)
        if converted_value != value:
            logger.info(f"[TARGET] UNIT CONVERSION: LLM conversion successful!")
            logger.info(
                f"[SUCCESS] UNIT CONVERSION RESULT: {value} {unit_str} -> {converted_value} {final_unit}"
            )
            logger.info(
                f"[STATS] UNIT CONVERSION: Conversion factor: {converted_value/value:.6f}"
            )
        else:
            logger.info(f"[SUCCESS] UNIT CONVERSION: LLM determined no conversion needed")
            logger.info(
                f"[SUCCESS] UNIT CONVERSION RESULT: {value} {unit_str} -> {converted_value} {final_unit}"
            )
        return converted_value, final_unit
    except Exception as e:
        logger.warning(
            f"[ERROR] UNIT CONVERSION: LLM conversion failed for {value} {unit_str}: {e}"
        )
        logger.info(f"[CONVERT] UNIT CONVERSION: Falling back to hard-coded conversion...")
        # Fallback to hard-coded conversion
        result = _fallback_convert_troponin_units(value, unit_str)
        logger.info(
            f"[SUCCESS] UNIT CONVERSION RESULT: {value} {unit_str} -> {result[0]} {result[1]} (fallback)"
        )
        return result


def _llm_convert_troponin_units(value: float, unit: str) -> tuple[float, str]:
    """Use LLM to convert troponin units to ng/mL.

    Args:
        value: The numeric value
        unit: The original unit

    Returns:
        Tuple of (converted_value, final_unit)
    """
    llm_client = LightenLLMClient()

    prompt = f"""You are a medical laboratory expert. Convert the following troponin measurement to ng/mL (nanograms per milliliter).

Input: {value} {unit}
Target unit: ng/mL

Please provide the conversion following these rules:
1. If the input unit is already ng/mL or equivalent, return the same value
2. Convert accurately using medical laboratory standards
3. Common troponin units include: ng/mL, μg/L, pg/mL, mg/L, ng/dL, μg/dL
4. Be aware that μg/L = ng/mL (same concentration)
5. 1 ng/mL = 1000 pg/mL
6. 1 mg/L = 1,000,000 ng/mL

Return your response as a JSON object with this exact format:
{{
    "converted_value": <numeric_value>,
    "final_unit": "ng/mL",
    "conversion_factor": <factor_used>,
    "explanation": "<brief explanation of conversion>"
}}

If the unit is unrecognizable or cannot be converted, set converted_value to the original value and explain in the explanation field."""

    try:
        logger.info(f"[LLM] CONVERSION: Sending prompt to LLM for {value} {unit}")
        response = llm_client.chat_completion([{"role": "user", "content": prompt}])
        response_text = response.strip()
        logger.info(f"[LLM] CONVERSION: Received response from LLM")

        # Parse JSON response
        if response_text.startswith("```json"):
            response_text = (
                response_text.replace("```json", "").replace("```", "").strip()
            )
            logger.info(f"[LLM] CONVERSION: Cleaned JSON response format")

        logger.info(f"[LLM] CONVERSION: Parsing JSON response...")
        result = json.loads(response_text)

        converted_value = float(result["converted_value"])
        final_unit = result["final_unit"]
        explanation = result.get("explanation", "")
        conversion_factor = result.get("conversion_factor", "N/A")

        logger.info(f"[SUCCESS] LLM CONVERSION SUCCESS:")
        logger.info(f"  [DATA] Original: {value} {unit}")
        logger.info(f"  [DATA] Converted: {converted_value} {final_unit}")
        logger.info(f"  [DATA] Conversion factor: {conversion_factor}")
        logger.info(f"  [DATA] Explanation: {explanation}")

        return converted_value, final_unit

    except json.JSONDecodeError as e:
        logger.error(f"[ERROR] LLM CONVERSION: JSON parsing failed: {e}")
        logger.error(f"[ERROR] LLM CONVERSION: Raw response: {response_text}")
        raise
    except Exception as e:
        logger.error(f"[ERROR] LLM CONVERSION: General error: {e}")
        raise


def _fallback_convert_troponin_units(value: float, unit: str) -> tuple[float, str]:
    """Fallback hard-coded unit conversion for troponin.

    Args:
        value: The numeric value
        unit: The original unit

    Returns:
        Tuple of (converted_value, final_unit)
    """
    unit_lower = unit.lower().strip()

    # Already in ng/mL - no conversion needed
    if unit_lower in ["ng/ml", "ng/dl", "nanogram/ml", "nanogram/dl"]:
        return value, "ng/mL"

    # Convert from μg/L (micrograms per liter) to ng/mL
    # 1 μg/L = 1 ng/mL (same concentration)
    if unit_lower in ["μg/l", "ug/l", "microgram/l", "mcg/l"]:
        logger.info(
            f"Fallback conversion: {value} {unit} -> {value} ng/mL (μg/L = ng/mL)"
        )
        return value, "ng/mL"

    # Convert from pg/mL (picograms per mL) to ng/mL
    # 1 ng/mL = 1000 pg/mL, so pg/mL ÷ 1000 = ng/mL
    if unit_lower in ["pg/ml", "pg/dl", "picogram/ml", "picogram/dl"]:
        converted = value / 1000.0
        logger.info(f"Fallback conversion: {value} {unit} -> {converted} ng/mL (÷1000)")
        return converted, "ng/mL"

    # Convert from mg/L (milligrams per liter) to ng/mL
    # 1 mg/L = 1000000 ng/mL
    if unit_lower in ["mg/l", "mg/dl", "milligram/l", "milligram/dl"]:
        converted = value * 1000.0
        logger.info(f"Fallback conversion: {value} {unit} -> {converted} ng/mL (×1000)")
        return converted, "ng/mL"

    # If unit is not recognized, log warning and return as-is
    logger.warning(f"Unknown troponin unit '{unit}', assuming ng/mL equivalent")
    return value, f"{unit} (assumed ng/mL)"


def compare_troponin_values(
    value1: float, unit1: str, value2: float, unit2: str
) -> Dict[str, any]:
    """Unit-aware comparison of troponin values.

    Args:
        value1: First value
        unit1: First value's unit
        value2: Second value
        unit2: Second value's unit

    Returns:
        Dictionary with comparison results
    """
    # Convert both values to ng/mL for comparison
    converted_value1, _ = convert_troponin_units(value1, unit1)
    converted_value2, _ = convert_troponin_units(value2, unit2)

    return {
        "value1_original": f"{value1} {unit1}",
        "value2_original": f"{value2} {unit2}",
        "value1_converted": f"{converted_value1} ng/mL",
        "value2_converted": f"{converted_value2} ng/mL",
        "comparison": (
            "greater"
            if converted_value1 > converted_value2
            else "less" if converted_value1 < converted_value2 else "equal"
        ),
        "difference": abs(converted_value1 - converted_value2),
        "ratio": (
            converted_value1 / converted_value2
            if converted_value2 != 0
            else float("inf")
        ),
    }


def is_above_troponin_threshold(
    value: float, unit: str, threshold: float = 0.014
) -> Dict[str, any]:
    """Unit-aware threshold comparison for troponin values.

    Args:
        value: The troponin value
        unit: The unit of the value
        threshold: The threshold in ng/mL (default 0.014 ng/mL)

    Returns:
        Dictionary with threshold comparison results
    """
    logger.info(f"[THRESHOLD] COMPARISON START: {value} {unit} vs {threshold} ng/mL")

    # Convert to standard units for comparison
    converted_value, final_unit = convert_troponin_units(value, unit)

    # Perform threshold comparison
    above_threshold = converted_value > threshold
    difference = converted_value - threshold
    fold_change = converted_value / threshold if threshold > 0 else float("inf")

    # Log detailed comparison results
    logger.info(f"[THRESHOLD] COMPARISON DETAILS:")
    logger.info(f"  [DATA] Original value: {value} {unit}")
    logger.info(f"  [DATA] Converted value: {converted_value} {final_unit}")
    logger.info(f"  [DATA] Threshold: {threshold} ng/mL")
    logger.info(f"  [DATA] Above threshold: {above_threshold}")
    logger.info(f"  [DATA] Difference: {difference:+.6f} ng/mL")
    
    # Handle NaN fold_change values
    if isinstance(fold_change, (int, float)) and not (fold_change != fold_change):  # Check for NaN
        logger.info(f"  [DATA] Fold change: {fold_change:.3f}x")
    else:
        logger.info(f"  [DATA] Fold change: N/A (invalid calculation)")

    if above_threshold:
        logger.info(
            f"[PASS] THRESHOLD COMPARISON: VALUE EXCEEDS THRESHOLD ({converted_value:.6f} > {threshold})"
        )
        if isinstance(fold_change, (int, float)) and not (fold_change != fold_change) and fold_change >= 5.0:
            logger.info(
                f"[CRITICAL] THRESHOLD COMPARISON: SIGNIFICANTLY ELEVATED (>=5x threshold)"
            )
        elif isinstance(fold_change, (int, float)) and not (fold_change != fold_change) and fold_change >= 2.0:
            logger.info(f"[WARNING] THRESHOLD COMPARISON: MODERATELY ELEVATED (>=2x threshold)")
        else:
            logger.info(f"[INFO] THRESHOLD COMPARISON: MILDLY ELEVATED (>1x threshold)")
    else:
        logger.info(
            f"[FAIL] THRESHOLD COMPARISON: VALUE BELOW THRESHOLD ({converted_value:.6f} <= {threshold})"
        )

    result = {
        "original_value": f"{value} {unit}",
        "converted_value": f"{converted_value} {final_unit}",
        "threshold": f"{threshold} ng/mL",
        "above_threshold": above_threshold,
        "difference_from_threshold": difference,
        "fold_change": fold_change,
    }

    logger.info(f"[SUCCESS] THRESHOLD COMPARISON COMPLETE: {above_threshold}")
    return result


def get_supported_troponin_units() -> dict[str, str]:
    """Get dictionary of supported troponin units and their descriptions.

    Returns:
        Dictionary mapping unit patterns to descriptions
    """
    return {
        "ng/mL": "Nanograms per milliliter (standard)",
        "μg/L": "Micrograms per liter (equivalent to ng/mL)",
        "pg/mL": "Picograms per milliliter (converted by ÷1000)",
        "mg/L": "Milligrams per liter (converted by ×1000)",
        "LLM-based": "Intelligent conversion using medical knowledge",
    }
