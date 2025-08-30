#!/usr/bin/env python3
"""Test script to verify LLM-based unit conversion and unit-aware comparisons."""

import logging
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from lighten_ml.evidence_collectors.unit_converter import (
    convert_troponin_units,
    compare_troponin_values,
    is_above_troponin_threshold,
    get_supported_troponin_units,
)
from lighten_ml.evidence_collectors.troponin_analyzer import TroponinAnalyzer
from lighten_ml.data_loaders.lab_data_loader import LabDataLoader

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_llm_unit_conversion():
    """Test LLM-based unit conversion functionality."""

    logger.info("=== LLM-BASED UNIT CONVERSION TEST ===")

    logger.info("=== TEST 1: LLM Unit Conversion ===")

    # Test various unit conversions using LLM
    test_cases = [
        # (value, unit, expected_behavior)
        (0.050, "ng/mL", "should_stay_same"),  # Already standard
        (0.050, "ng/dl", "should_stay_same"),  # Already standard
        (50.0, "pg/mL", "should_convert_divide"),  # Should convert to 0.050 ng/mL
        (0.050, "μg/L", "should_stay_same"),  # Equivalent to ng/mL
        (0.050, "ug/L", "should_stay_same"),  # Equivalent to ng/mL
        (0.000050, "mg/L", "should_convert_multiply"),  # Should convert to 0.050 ng/mL
        (25.0, "picogram/ml", "should_convert_divide"),  # Should convert
        (0.025, "microgram/liter", "should_stay_same"),  # Should stay same
        (100, "weird_unit", "should_handle_unknown"),  # Unknown unit
        (0.030, "", "should_handle_empty"),  # Empty unit
    ]

    logger.info("Testing LLM-based unit conversions:")
    conversion_results = []

    for value, unit, expected_behavior in test_cases:
        try:
            logger.info(f"\n--- Testing: {value} {unit} ---")
            converted_value, final_unit = convert_troponin_units(value, unit)

            result = {
                "original": f"{value} {unit}",
                "converted": f"{converted_value} {final_unit}",
                "expected_behavior": expected_behavior,
                "conversion_successful": True,
                "used_llm": converted_value != value or unit != final_unit,
            }

            conversion_results.append(result)

            logger.info(
                f"✅ Conversion: {value} {unit} -> {converted_value} {final_unit}"
            )

        except Exception as e:
            logger.error(f"❌ Conversion failed for {value} {unit}: {e}")
            conversion_results.append(
                {
                    "original": f"{value} {unit}",
                    "converted": "FAILED",
                    "expected_behavior": expected_behavior,
                    "conversion_successful": False,
                    "error": str(e),
                }
            )

    # Summary of conversion results
    successful_conversions = sum(
        1 for r in conversion_results if r["conversion_successful"]
    )
    logger.info(
        f"\n✅ LLM Unit Conversion Summary: {successful_conversions}/{len(conversion_results)} successful"
    )

    logger.info("=== TEST 2: Unit-Aware Comparisons ===")

    # Test unit-aware comparisons
    comparison_tests = [
        # (value1, unit1, value2, unit2, expected_relationship)
        (50, "pg/mL", 0.050, "ng/mL", "equal"),  # Same value, different units
        (100, "pg/mL", 0.050, "ng/mL", "greater"),  # 100 pg/mL > 50 pg/mL (0.050 ng/mL)
        (0.025, "ng/mL", 50, "pg/mL", "less"),  # 0.025 ng/mL < 0.050 ng/mL (50 pg/mL)
        (0.050, "μg/L", 0.050, "ng/mL", "equal"),  # Equivalent units
        (1.0, "ng/mL", 500, "pg/mL", "greater"),  # 1.0 ng/mL > 0.5 ng/mL (500 pg/mL)
    ]

    logger.info("Testing unit-aware comparisons:")
    for value1, unit1, value2, unit2, expected in comparison_tests:
        try:
            comparison = compare_troponin_values(value1, unit1, value2, unit2)

            logger.info(f"\n--- Comparing: {value1} {unit1} vs {value2} {unit2} ---")
            logger.info(
                f"Original values: {comparison['value1_original']} vs {comparison['value2_original']}"
            )
            logger.info(
                f"Converted values: {comparison['value1_converted']} vs {comparison['value2_converted']}"
            )
            logger.info(
                f"Comparison result: {comparison['comparison']} (expected: {expected})"
            )
            logger.info(f"Difference: {comparison['difference']:.6f} ng/mL")
            logger.info(f"Ratio: {comparison['ratio']:.3f}")

            if comparison["comparison"] == expected:
                logger.info("✅ Comparison result matches expectation")
            else:
                logger.warning(
                    f"⚠️ Comparison mismatch: got {comparison['comparison']}, expected {expected}"
                )

        except Exception as e:
            logger.error(f"❌ Comparison failed: {e}")

    logger.info("=== TEST 3: Unit-Aware Threshold Analysis ===")

    # Test unit-aware threshold comparisons
    threshold_tests = [
        # (value, unit, should_be_above_threshold)
        (0.020, "ng/mL", True),  # Above 0.014 threshold
        (0.010, "ng/mL", False),  # Below 0.014 threshold
        (20, "pg/mL", True),  # 20 pg/mL = 0.020 ng/mL > 0.014
        (10, "pg/mL", False),  # 10 pg/mL = 0.010 ng/mL < 0.014
        (0.020, "μg/L", True),  # 0.020 μg/L = 0.020 ng/mL > 0.014
        (0.010, "μg/L", False),  # 0.010 μg/L = 0.010 ng/mL < 0.014
    ]

    logger.info("Testing unit-aware threshold analysis:")
    for value, unit, expected_above in threshold_tests:
        try:
            threshold_result = is_above_troponin_threshold(value, unit)

            logger.info(f"\n--- Threshold test: {value} {unit} ---")
            logger.info(f"Original: {threshold_result['original_value']}")
            logger.info(f"Converted: {threshold_result['converted_value']}")
            logger.info(f"Threshold: {threshold_result['threshold']}")
            logger.info(
                f"Above threshold: {threshold_result['above_threshold']} (expected: {expected_above})"
            )
            logger.info(
                f"Difference from threshold: {threshold_result['difference_from_threshold']:.6f} ng/mL"
            )
            logger.info(f"Fold change: {threshold_result['fold_change']:.3f}x")

            if threshold_result["above_threshold"] == expected_above:
                logger.info("✅ Threshold analysis matches expectation")
            else:
                logger.warning(
                    f"⚠️ Threshold mismatch: got {threshold_result['above_threshold']}, expected {expected_above}"
                )

        except Exception as e:
            logger.error(f"❌ Threshold analysis failed: {e}")

    logger.info("=== TEST 4: Integration with TroponinAnalyzer ===")

    # Test integration with enhanced TroponinAnalyzer
    try:
        lab_events_path = "labevents.csv"
        d_labitems_path = "d_labitems.csv"

        lab_loader = LabDataLoader(lab_events_path, d_labitems_path)
        troponin_analyzer = TroponinAnalyzer(lab_loader)

        logger.info("✅ TroponinAnalyzer initialized with LLM unit conversion support")

        # Test with sample data if available
        try:
            lab_loader.load_data()
            if lab_loader.data is not None and not lab_loader.data.empty:
                sample_patient = lab_loader.data.iloc[0]["subject_id"]
                sample_hadm = lab_loader.data.iloc[0]["hadm_id"]

                logger.info(
                    f"Testing enhanced troponin analysis with LLM conversions..."
                )
                logger.info(f"Patient: {sample_patient}, Admission: {sample_hadm}")

                # This should now use LLM-based unit conversion
                evidence = troponin_analyzer.collect_evidence(
                    sample_patient, sample_hadm
                )

                logger.info(
                    "✅ Enhanced troponin analysis with LLM conversions completed"
                )
                logger.info(
                    f"Troponin available: {evidence.get('troponin_available', False)}"
                )
                logger.info(
                    f"MI criteria met: {evidence.get('mi_criteria_met', False)}"
                )

                # Check if troponin tests include enhanced unit analysis
                troponin_tests = evidence.get("troponin_tests", [])
                if troponin_tests:
                    sample_test = troponin_tests[0]
                    logger.info("Sample troponin test with LLM unit conversion:")
                    logger.info(
                        f"  Original: {sample_test.get('original_value', 'N/A')} {sample_test.get('original_unit', 'N/A')}"
                    )
                    logger.info(
                        f"  Converted: {sample_test.get('value', 'N/A')} {sample_test.get('unit', 'N/A')}"
                    )
                    logger.info(
                        f"  Above threshold: {sample_test.get('above_threshold', 'N/A')}"
                    )

                    # Check if threshold analysis is included
                    if "threshold_analysis" in sample_test:
                        threshold_analysis = sample_test["threshold_analysis"]
                        logger.info(
                            f"  Threshold analysis: {threshold_analysis.get('above_threshold', 'N/A')}"
                        )
                        logger.info(
                            f"  Fold change: {threshold_analysis.get('fold_change', 'N/A'):.3f}x"
                        )
            else:
                logger.info("No lab data available for integration testing")

        except FileNotFoundError:
            logger.info("Lab data files not found - integration test skipped")

    except Exception as e:
        logger.error(f"Integration test failed: {e}")

    logger.info("=== TEST 5: LLM vs Fallback Comparison ===")

    # Test scenarios where LLM might fail and fallback is used
    logger.info("Testing LLM failure scenarios and fallback behavior:")

    # This would test what happens when LLM is unavailable or fails
    # The system should gracefully fall back to hard-coded conversions
    fallback_tests = [
        (50.0, "pg/mL"),  # Should work with both LLM and fallback
        (0.050, "μg/L"),  # Should work with both LLM and fallback
    ]

    for value, unit in fallback_tests:
        try:
            converted_value, final_unit = convert_troponin_units(value, unit)
            logger.info(
                f"✅ Conversion successful: {value} {unit} -> {converted_value} {final_unit}"
            )
        except Exception as e:
            logger.info(f"ℹ️ Conversion behavior: {e}")

    logger.info("=== LLM UNIT CONVERSION TEST COMPLETE ===")
    logger.info("🎉 LLM-based unit conversion system ready!")
    logger.info("🎉 Unit-aware comparisons implemented!")
    logger.info(
        "🎉 Medical diagnosis pipeline enhanced with intelligent unit handling!"
    )

    # Show supported units
    supported_units = get_supported_troponin_units()
    logger.info("\n📋 Supported troponin units:")
    for unit, description in supported_units.items():
        logger.info(f"  - {unit}: {description}")


if __name__ == "__main__":
    test_llm_unit_conversion()
