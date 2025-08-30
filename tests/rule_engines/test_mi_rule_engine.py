import unittest
from datetime import datetime

from lighten_ml.rule_engines.mi_rule_engine import MIRuleEngine, MIRuleEngineConfig


class TestMIRuleEngine(unittest.TestCase):

    def setUp(self):
        """Set up a default rule engine for tests."""
        self.config = MIRuleEngineConfig()
        self.engine = MIRuleEngine(self.config)

    def test_mi_with_rise_fall_pattern(self):
        """Test MI detection with a clear troponin rise/fall pattern."""
        evidence = {
            "troponin": {
                "mi_criteria_met": True,
                "criteria_details": {"criteria": "rise and fall pattern"},
            },
            "clinical": {"ischemic_symptoms_present": False},
        }

        result = self.engine.evaluate(evidence)

        self.assertTrue(result.passed)
        self.assertIn("biomarker and ischemia evidence", result.details["summary"])

    def test_mi_with_single_troponin_and_ischemia(self):
        """Test MI with single high troponin and ischemic symptoms."""
        evidence = {
            "troponin": {
                "mi_criteria_met": True,
                "criteria_details": {"criteria": "Single elevated troponin"},
            },
            "clinical": {"ischemic_symptoms_present": True},
        }

        result = self.engine.evaluate(evidence)

        self.assertTrue(result.passed)
        self.assertIn(
            "single elevated troponin with ischemia", result.details["summary"]
        )

    def test_no_mi_with_single_troponin_no_ischemia(self):
        """Test no MI with single high troponin but no ischemic evidence."""
        evidence = {
            "troponin": {
                "mi_criteria_met": True,
                "criteria_details": {"criteria": "Single elevated troponin"},
            },
            "clinical": {"ischemic_symptoms_present": False},
            "ecg": {"ecg_findings": []},
        }

        # Default config requires ischemia for single troponin
        self.config.require_ischemia_for_single_troponin = True
        self.engine = MIRuleEngine(self.config)

        result = self.engine.evaluate(evidence)

        self.assertFalse(result.passed)

    def test_no_mi_with_ischemia_no_biomarker(self):
        """Test no MI with ischemic evidence but no troponin elevation."""
        evidence = {
            "troponin": {
                "mi_criteria_met": False,
                "criteria_details": {"reason": "Values are normal"},
            },
            "clinical": {"ischemic_symptoms_present": True},
        }

        result = self.engine.evaluate(evidence)

        self.assertFalse(result.passed)

    def test_config_require_both_criteria(self):
        """Test configuration to require both biomarker and ischemia criteria."""
        evidence = {
            "troponin": {
                "mi_criteria_met": True,
                "criteria_details": {"criteria": "rise and fall pattern"},
            },
            "clinical": {"ischemic_symptoms_present": False},
        }

        # Configure engine to require both A and B
        self.config.require_both_criteria = True
        self.engine = MIRuleEngine(self.config)

        result = self.engine.evaluate(evidence)

        self.assertFalse(result.passed)  # Fails because ischemia is missing


if __name__ == "__main__":
    unittest.main()
