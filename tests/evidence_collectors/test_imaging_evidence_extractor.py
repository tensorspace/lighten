import unittest
from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd

from lighten_ml.evidence_collectors.imaging_evidence_extractor import (
    ImagingEvidenceExtractor,
)


class TestImagingEvidenceExtractor(unittest.TestCase):

    def setUp(self):
        """Set up mock data loader and LLM client for tests."""
        self.mock_notes_loader = MagicMock()
        self.mock_llm_client = MagicMock()
        self.extractor = ImagingEvidenceExtractor(
            self.mock_notes_loader, llm_client=self.mock_llm_client
        )

    def test_regex_extraction(self):
        """Test imaging evidence extraction using regex."""
        patient_id = "401"
        hadm_id = "adm_img_1"

        notes = pd.DataFrame(
            {
                "patient_id": [patient_id],
                "hadm_id": [hadm_id],
                "note_type": ["Echo"],
                "charttime": [datetime(2023, 4, 1, 14, 0)],
                "text": [
                    "Echocardiogram shows a new wall motion abnormality in the anterior region."
                ],
            }
        )

        self.mock_notes_loader.get_patient_notes.return_value = notes
        self.mock_llm_client.enabled = False

        evidence = self.extractor.collect_evidence(patient_id, hadm_id)

        self.assertEqual(evidence["metadata"]["extraction_mode"], "regex")
        self.assertTrue(evidence["wall_motion_abnormalities"])
        self.assertEqual(len(evidence["imaging_findings"]), 1)
        self.assertEqual(
            evidence["imaging_findings"][0]["finding"], "Wall Motion Abnormality"
        )

    def test_llm_extraction(self):
        """Test imaging evidence extraction using LLM."""
        patient_id = "402"
        hadm_id = "adm_img_2"

        notes = pd.DataFrame(
            {
                "patient_id": [patient_id],
                "hadm_id": [hadm_id],
                "note_type": ["Radiology"],
                "charttime": [datetime(2023, 4, 5, 16, 0)],
                "text": [
                    "Echo reveals severe hypokinesis of the apex. This is an acute finding."
                ],
            }
        )

        llm_output = {
            "imaging_findings": [
                {
                    "finding": "Wall Motion Abnormality",
                    "context": "hypokinesis of the apex",
                    "is_new": True,
                }
            ]
        }

        self.mock_notes_loader.get_patient_notes.return_value = notes
        self.mock_llm_client.enabled = True
        self.mock_llm_client.extract_json.return_value = llm_output

        evidence = self.extractor.collect_evidence(patient_id, hadm_id)

        self.assertEqual(evidence["metadata"]["extraction_mode"], "llm")
        self.assertTrue(evidence["wall_motion_abnormalities"])
        self.assertEqual(len(evidence["imaging_findings"]), 1)
        self.assertTrue(evidence["imaging_findings"][0]["is_new"])

    def test_patient_centric_logging_in_regex_mode(self):
        """Verify patient and admission IDs are in logs (Regex mode)."""
        patient_id = "log_img_1"
        hadm_id = "log_adm_img_1"
        notes = pd.DataFrame({"text": ["WMA noted."]})

        self.mock_notes_loader.get_patient_notes.return_value = notes
        self.mock_llm_client.enabled = False

        with self.assertLogs('lighten_ml.evidence_collectors.imaging_evidence_extractor', level='INFO') as cm:
            self.extractor.collect_evidence(patient_id, hadm_id)
            self.assertTrue(any(f"[{patient_id}][{hadm_id}]" in msg for msg in cm.output))

    def test_patient_centric_logging_in_llm_mode(self):
        """Verify patient and admission IDs are in logs (LLM mode)."""
        patient_id = "log_img_2"
        hadm_id = "log_adm_img_2"
        notes = pd.DataFrame({"text": ["WMA noted."]})
        llm_output = {"imaging_findings": []}

        self.mock_notes_loader.get_patient_notes.return_value = notes
        self.mock_llm_client.enabled = True
        self.mock_llm_client.extract_json.return_value = llm_output

        with self.assertLogs('lighten_ml.evidence_collectors.imaging_evidence_extractor', level='INFO') as cm:
            self.extractor.collect_evidence(patient_id, hadm_id)
            self.assertTrue(any(f"[{patient_id}][{hadm_id}]" in msg for msg in cm.output))


if __name__ == "__main__":
    unittest.main()
