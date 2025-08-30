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
    if not unit or pd.isna(unit):
        logger.warning(f"No unit provided for troponin value {value}, assuming ng/mL")
        return value, "ng/mL"

    unit_str = str(unit).strip()

    # Quick check for already standard units
    if unit_str.lower() in ["ng/ml", "ng/dl", "nanogram/ml", "nanogram/dl"]:
        return value, "ng/mL"

    # Use LLM for intelligent unit conversion
    try:
        converted_value, final_unit = _llm_convert_troponin_units(value, unit_str)
        if converted_value != value:
            logger.info(
                f"LLM converted troponin: {value} {unit_str} -> {converted_value} {final_unit}"
            )
        return converted_value, final_unit
    except Exception as e:
        logger.warning(f"LLM conversion failed for {value} {unit_str}: {e}")
        # Fallback to hard-coded conversion
        return _fallback_convert_troponin_units(value, unit_str)


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
        response = llm_client.chat_completion([{"role": "user", "content": prompt}])
        response_text = response.strip()

        # Parse JSON response
        if response_text.startswith("```json"):
            response_text = (
                response_text.replace("```json", "").replace("```", "").strip()
            )

        result = json.loads(response_text)

        converted_value = float(result["converted_value"])
        final_unit = result["final_unit"]
        explanation = result.get("explanation", "")

        logger.info(
            f"LLM unit conversion: {value} {unit} -> {converted_value} {final_unit} ({explanation})"
        )

        return converted_value, final_unit

    except Exception as e:
        logger.error(f"Error in LLM unit conversion: {e}")
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
    converted_value, final_unit = convert_troponin_units(value, unit)

    above_threshold = converted_value > threshold

    return {
        "original_value": f"{value} {unit}",
        "converted_value": f"{converted_value} {final_unit}",
        "threshold": f"{threshold} ng/mL",
        "above_threshold": above_threshold,
        "difference_from_threshold": converted_value - threshold,
        "fold_change": converted_value / threshold if threshold > 0 else float("inf"),
    }


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
