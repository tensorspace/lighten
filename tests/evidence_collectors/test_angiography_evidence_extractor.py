import unittest
from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd

from lighten_ml.evidence_collectors.angiography_evidence_extractor import (
    AngiographyEvidenceExtractor,
)


class TestAngiographyEvidenceExtractor(unittest.TestCase):

    def setUp(self):
        """Set up mock data loader and LLM client for tests."""
        self.mock_notes_loader = MagicMock()
        self.mock_llm_client = MagicMock()
        self.extractor = AngiographyEvidenceExtractor(
            self.mock_notes_loader, llm_client=self.mock_llm_client
        )

    def test_regex_extraction(self):
        """Test angiography evidence extraction using regex."""
        patient_id = "501"
        hadm_id = "adm_ang_1"

        notes = pd.DataFrame(
            {
                "patient_id": [patient_id],
                "hadm_id": [hadm_id],
                "note_type": ["Cardiac Cath"],
                "charttime": [datetime(2023, 5, 1, 11, 0)],
                "text": [
                    "Cardiac cath reveals a large thrombus in the left anterior descending artery."
                ],
            }
        )

        self.mock_notes_loader.get_patient_notes.return_value = notes
        self.mock_llm_client.enabled = False

        evidence = self.extractor.collect_evidence(patient_id, hadm_id)

        self.assertEqual(evidence["metadata"]["extraction_mode"], "regex")
        self.assertTrue(evidence["thrombus_present"])
        self.assertEqual(len(evidence["angiography_findings"]), 1)
        self.assertEqual(
            evidence["angiography_findings"][0]["finding"], "Intracoronary Thrombus"
        )

    def test_llm_extraction(self):
        """Test angiography evidence extraction using LLM."""
        patient_id = "502"
        hadm_id = "adm_ang_2"

        notes = pd.DataFrame(
            {
                "patient_id": [patient_id],
                "hadm_id": [hadm_id],
                "note_type": ["Cardiac Cath"],
                "charttime": [datetime(2023, 5, 10, 15, 0)],
                "text": ["Angiography shows a thrombotic occlusion of the RCA."],
            }
        )

        llm_output = {
            "angiography_findings": [
                {
                    "finding": "Intracoronary Thrombus",
                    "context": "thrombotic occlusion of the RCA",
                    "vessel": "RCA",
                }
            ]
        }

        self.mock_notes_loader.get_patient_notes.return_value = notes
        self.mock_llm_client.enabled = True
        self.mock_llm_client.extract_json.return_value = llm_output

        evidence = self.extractor.collect_evidence(patient_id, hadm_id)

        self.assertEqual(evidence["metadata"]["extraction_mode"], "llm")
        self.assertTrue(evidence["thrombus_present"])
        self.assertEqual(len(evidence["angiography_findings"]), 1)
        self.assertEqual(evidence["angiography_findings"][0]["vessel"], "RCA")


if __name__ == "__main__":
    unittest.main()
