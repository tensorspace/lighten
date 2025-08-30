import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pandas as pd

from lighten_ml.evidence_collectors.troponin_analyzer import TroponinAnalyzer


class TestTroponinAnalyzer(unittest.TestCase):

    def setUp(self):
        """Set up mock data loader for tests."""
        self.mock_lab_loader = MagicMock()
        self.analyzer = TroponinAnalyzer(self.mock_lab_loader, time_window_hours=72)

    def test_rise_and_fall_pattern(self):
        """Test detection of a classic rise and fall pattern."""
        patient_id = "101"
        hadm_id = "adm1"

        # Mock data representing a rise and fall
        troponin_data = pd.DataFrame(
            {
                "patient_id": [patient_id] * 3,
                "hadm_id": [hadm_id] * 3,
                "timestamp": [
                    datetime(2023, 1, 1, 6, 0),
                    datetime(2023, 1, 1, 12, 0),  # Peak
                    datetime(2023, 1, 1, 18, 0),
                ],
                "value": [0.5, 2.5, 1.0],
                "valueuom": ["ng/mL"] * 3,
                "label": ["Troponin T"] * 3,
                "ref_range_upper": [0.04] * 3,
            }
        )

        self.mock_lab_loader.get_patient_labs.return_value = troponin_data

        evidence = self.analyzer.collect_evidence(patient_id, hadm_id)

        self.assertTrue(evidence["mi_criteria_met"])
        self.assertEqual(
            evidence["criteria_details"]["criteria"], "rise and fall pattern"
        )
        self.assertEqual(len(evidence["troponin_tests"]), 3)
        self.assertAlmostEqual(evidence["criteria_details"]["max_value"], 2.5)

    def test_single_elevated_troponin(self):
        """Test detection of a single elevated troponin value."""
        patient_id = "102"
        hadm_id = "adm2"

        troponin_data = pd.DataFrame(
            {
                "patient_id": [patient_id],
                "hadm_id": [hadm_id],
                "timestamp": [datetime(2023, 1, 2, 8, 0)],
                "value": [1.5],
                "valueuom": ["ng/mL"],
                "label": ["Troponin T"],
                "ref_range_upper": [0.04],
            }
        )

        self.mock_lab_loader.get_patient_labs.return_value = troponin_data

        evidence = self.analyzer.collect_evidence(patient_id, hadm_id)

        self.assertTrue(evidence["mi_criteria_met"])
        self.assertEqual(
            evidence["criteria_details"]["criteria"], "Single elevated troponin"
        )
        self.assertAlmostEqual(evidence["criteria_details"]["max_value"], 1.5)

    def test_no_significant_change(self):
        """Test scenario with elevated but stable troponin values."""
        patient_id = "103"
        hadm_id = "adm3"

        troponin_data = pd.DataFrame(
            {
                "patient_id": [patient_id] * 3,
                "hadm_id": [hadm_id] * 3,
                "timestamp": [
                    datetime(2023, 1, 3, 6, 0),
                    datetime(2023, 1, 3, 12, 0),
                    datetime(2023, 1, 3, 18, 0),
                ],
                "value": [0.6, 0.65, 0.58],
                "valueuom": ["ng/mL"] * 3,
                "label": ["Troponin T"] * 3,
                "ref_range_upper": [0.04] * 3,
            }
        )

        self.mock_lab_loader.get_patient_labs.return_value = troponin_data

        evidence = self.analyzer.collect_evidence(patient_id, hadm_id)

        self.assertFalse(evidence["mi_criteria_met"])
        self.assertIn(
            "No significant rise/fall pattern", evidence["criteria_details"]["reason"]
        )

    def test_time_window_exclusion(self):
        """Test that values outside the time window are ignored."""
        patient_id = "104"
        hadm_id = "adm4"

        troponin_data = pd.DataFrame(
            {
                "patient_id": [patient_id] * 2,
                "hadm_id": [hadm_id] * 2,
                "timestamp": [
                    datetime(2023, 1, 4, 6, 0),
                    datetime(2023, 1, 8, 12, 0),  # 4 days later
                ],
                "value": [0.5, 2.0],
                "valueuom": ["ng/mL"] * 2,
                "label": ["Troponin T"] * 2,
                "ref_range_upper": [0.04] * 2,
            }
        )

        self.mock_lab_loader.get_patient_labs.return_value = troponin_data

        # With a 72-hour window, the second value should be ignored for pattern analysis
        evidence = self.analyzer.collect_evidence(patient_id, hadm_id)

        self.assertTrue(evidence["mi_criteria_met"])
        self.assertEqual(
            evidence["criteria_details"]["criteria"], "Single elevated troponin"
        )
        self.assertEqual(
            len(evidence["troponin_tests"]), 2
        )  # Both are still included in the list


if __name__ == "__main__":
    unittest.main()
