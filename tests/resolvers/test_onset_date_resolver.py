import unittest
from datetime import datetime

import pandas as pd

from lighten_ml.resolvers.onset_date_resolver import OnsetDateResolver


class TestOnsetDateResolver(unittest.TestCase):

    def setUp(self):
        """Set up the onset date resolver."""
        self.resolver = OnsetDateResolver()
        self.admission_time = pd.Timestamp("2023-01-15 12:00:00")

    def test_priority_1_symptom_onset(self):
        """Test that symptom onset date is prioritized."""
        evidence = {
            "clinical": {"symptoms": [{"onset_time": "2023-01-15 08:00:00"}]},
            "ecg": {
                "ecg_findings": [
                    {"charttime": "2023-01-15 09:00:00", "mi_related": True}
                ]
            },
            "troponin": {
                "troponin_tests": [
                    {"timestamp": "2023-01-15 10:00:00", "above_threshold": True}
                ]
            },
        }
        result = self.resolver.resolve(evidence, self.admission_time)
        self.assertEqual(result["rationale"], "Symptom onset date")
        self.assertEqual(result["onset_date"], datetime(2023, 1, 15, 8, 0).isoformat())

    def test_priority_2_ecg_onset(self):
        """Test that ECG date is used when no symptom date is available."""
        evidence = {
            "clinical": {"symptoms": []},
            "ecg": {
                "ecg_findings": [
                    {"charttime": "2023-01-15 09:00:00", "mi_related": True}
                ]
            },
            "troponin": {
                "troponin_tests": [
                    {"timestamp": "2023-01-15 10:00:00", "above_threshold": True}
                ]
            },
        }
        result = self.resolver.resolve(evidence, self.admission_time)
        self.assertEqual(result["rationale"], "First abnormal ECG date")
        self.assertEqual(result["onset_date"], datetime(2023, 1, 15, 9, 0).isoformat())

    def test_priority_3_troponin_onset(self):
        """Test that troponin date is used when it's the earliest available."""
        evidence = {
            "clinical": {"symptoms": []},
            "ecg": {"ecg_findings": []},
            "troponin": {
                "troponin_tests": [
                    {"timestamp": "2023-01-15 10:00:00", "above_threshold": True}
                ]
            },
        }
        result = self.resolver.resolve(evidence, self.admission_time)
        self.assertEqual(result["rationale"], "First elevated troponin date")
        self.assertEqual(result["onset_date"], datetime(2023, 1, 15, 10, 0).isoformat())

    def test_priority_4_diagnosis_date(self):
        """Test that diagnosis date is used when available."""
        evidence = {
            "clinical": {
                "symptoms": [],
                "diagnoses": [{"diagnosis_date": "2023-01-16"}],
            },
            "ecg": {"ecg_findings": []},
            "troponin": {"troponin_tests": []},
        }
        result = self.resolver.resolve(evidence, self.admission_time)
        self.assertEqual(result["rationale"], "Clinical diagnosis date")
        self.assertEqual(result["onset_date"], datetime(2023, 1, 16).isoformat())

    def test_priority_5_admission_date_fallback(self):
        """Test that admission date is used as a fallback."""
        evidence = {
            "clinical": {"symptoms": [], "diagnoses": []},
            "ecg": {"ecg_findings": []},
            "troponin": {"troponin_tests": []},
        }
        result = self.resolver.resolve(evidence, self.admission_time)
        self.assertEqual(result["rationale"], "Hospital admission date")
        self.assertEqual(result["onset_date"], self.admission_time.isoformat())

    def test_no_date_found(self):
        """Test that None is returned when no date can be found."""
        evidence = {}
        result = self.resolver.resolve(evidence, admission_time=None)
        self.assertIsNone(result["onset_date"])
        self.assertEqual(
            result["rationale"], "No definitive onset date could be determined"
        )


if __name__ == "__main__":
    unittest.main()
