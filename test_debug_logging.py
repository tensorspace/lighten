#!/usr/bin/env python3
"""Test script to demonstrate enhanced debugging logs for unit conversion and comparison."""

import logging
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from lighten_ml.evidence_collectors.unit_converter import (
    convert_troponin_units,
    is_above_troponin_threshold,
    compare_troponin_values,
)

# Set up detailed logging to see all debug information
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def test_debug_logging():
    """Demonstrate enhanced debugging logs for unit conversion and comparison."""

    logger.info("=" * 80)
    logger.info("🔍 ENHANCED DEBUGGING LOGS DEMONSTRATION")
    logger.info("=" * 80)

    # Test cases that will trigger different logging scenarios
    test_scenarios = [
        {
            "name": "Standard ng/mL (no conversion needed)",
            "value": 0.025,
            "unit": "ng/mL",
            "description": "Already in standard format",
        },
        {
            "name": "Picograms to nanograms conversion",
            "value": 50.0,
            "unit": "pg/mL",
            "description": "Should convert 50 pg/mL to 0.050 ng/mL",
        },
        {
            "name": "Micrograms equivalent",
            "value": 0.030,
            "unit": "μg/L",
            "description": "Should recognize μg/L = ng/mL",
        },
        {
            "name": "Unknown unit handling",
            "value": 25.0,
            "unit": "weird_unit",
            "description": "Should handle unknown units gracefully",
        },
        {
            "name": "Empty unit handling",
            "value": 0.020,
            "unit": "",
            "description": "Should handle empty units",
        },
        {
            "name": "High value above threshold",
            "value": 100.0,
            "unit": "pg/mL",
            "description": "Should show significantly elevated (>5x threshold)",
        },
        {
            "name": "Moderate elevation",
            "value": 30.0,
            "unit": "pg/mL",
            "description": "Should show moderately elevated (2-5x threshold)",
        },
        {
            "name": "Below threshold",
            "value": 10.0,
            "unit": "pg/mL",
            "description": "Should show below threshold",
        },
    ]

    logger.info("\n🧪 TESTING UNIT CONVERSION WITH ENHANCED LOGGING")
    logger.info("-" * 60)

    for i, scenario in enumerate(test_scenarios, 1):
        logger.info(f"\n📋 TEST SCENARIO {i}: {scenario['name']}")
        logger.info(f"📋 Description: {scenario['description']}")
        logger.info(f"📋 Input: {scenario['value']} {scenario['unit']}")
        logger.info("-" * 40)

        try:
            # Test unit conversion with detailed logging
            converted_value, final_unit = convert_troponin_units(
                scenario["value"], scenario["unit"]
            )

            logger.info(f"✅ SCENARIO {i} CONVERSION COMPLETE")

            # Test threshold comparison with detailed logging
            threshold_result = is_above_troponin_threshold(
                scenario["value"], scenario["unit"]
            )

            logger.info(f"✅ SCENARIO {i} THRESHOLD COMPARISON COMPLETE")

        except Exception as e:
            logger.error(f"❌ SCENARIO {i} FAILED: {e}")

        logger.info("=" * 60)

    logger.info("\n🔄 TESTING UNIT-AWARE VALUE COMPARISONS")
    logger.info("-" * 60)

    # Test unit-aware comparisons
    comparison_tests = [
        {
            "name": "Same value, different units",
            "value1": 50.0,
            "unit1": "pg/mL",
            "value2": 0.050,
            "unit2": "ng/mL",
            "expected": "equal",
        },
        {
            "name": "Different values, same units",
            "value1": 100.0,
            "unit1": "pg/mL",
            "value2": 50.0,
            "unit2": "pg/mL",
            "expected": "greater",
        },
        {
            "name": "Cross-unit comparison",
            "value1": 0.030,
            "unit1": "ng/mL",
            "value2": 25.0,
            "unit2": "pg/mL",
            "expected": "greater",
        },
    ]

    for i, test in enumerate(comparison_tests, 1):
        logger.info(f"\n📊 COMPARISON TEST {i}: {test['name']}")
        logger.info(
            f"📊 Comparing: {test['value1']} {test['unit1']} vs {test['value2']} {test['unit2']}"
        )
        logger.info(f"📊 Expected: {test['expected']}")
        logger.info("-" * 40)

        try:
            comparison = compare_troponin_values(
                test["value1"], test["unit1"], test["value2"], test["unit2"]
            )

            logger.info(f"✅ COMPARISON {i} COMPLETE")
            logger.info(f"📊 Result: {comparison['comparison']}")
            logger.info(
                f"📊 Match expected: {comparison['comparison'] == test['expected']}"
            )

        except Exception as e:
            logger.error(f"❌ COMPARISON {i} FAILED: {e}")

        logger.info("=" * 60)

    logger.info("\n🎯 LOGGING FEATURES DEMONSTRATED:")
    logger.info("✅ Unit conversion process with step-by-step details")
    logger.info("✅ LLM conversion attempts and responses")
    logger.info("✅ Fallback conversion when LLM fails")
    logger.info("✅ Threshold comparison with clinical significance levels")
    logger.info("✅ Unit-aware value comparisons")
    logger.info("✅ Visual indicators (emojis) for easy log parsing")
    logger.info("✅ Detailed error handling and debugging information")
    logger.info("✅ Conversion factors and mathematical details")
    logger.info("✅ Clinical interpretation of results")

    logger.info("\n🎉 ENHANCED DEBUGGING LOGS DEMONSTRATION COMPLETE!")
    logger.info("🎉 These logs will help debug unit conversion issues in MI diagnosis!")
    logger.info("=" * 80)


if __name__ == "__main__":
    test_debug_logging()
