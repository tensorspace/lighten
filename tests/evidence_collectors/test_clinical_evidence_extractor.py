import unittest
from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd

from lighten_ml.evidence_collectors.clinical_evidence_extractor import (
    ClinicalEvidenceExtractor,
)


class TestClinicalEvidenceExtractor(unittest.TestCase):

    def setUp(self):
        """Set up mock data loader and LLM client for tests."""
        self.mock_notes_loader = MagicMock()
        self.mock_llm_client = MagicMock()
        self.extractor = ClinicalEvidenceExtractor(
            self.mock_notes_loader, llm_client=self.mock_llm_client
        )

    def test_regex_symptom_extraction(self):
        """Test symptom extraction using regex."""
        patient_id = "301"
        hadm_id = "adm_clin_1"

        notes = pd.DataFrame(
            {
                "patient_id": [patient_id],
                "hadm_id": [hadm_id],
                "note_type": ["Nursing"],
                "charttime": [datetime(2023, 3, 1, 12, 0)],
                "text": [
                    "Patient complains of crushing chest pain and shortness of breath."
                ],
            }
        )

        self.mock_notes_loader.get_patient_notes.return_value = notes
        self.mock_llm_client.enabled = False

        evidence = self.extractor.collect_evidence(patient_id, hadm_id)

        self.assertEqual(evidence["metadata"]["extraction_mode"], "regex")
        self.assertTrue(evidence["ischemic_symptoms_present"])
        self.assertEqual(len(evidence["symptoms"]), 2)
        self.assertTrue(any(s["symptom"] == "Chest Pain" for s in evidence["symptoms"]))

    def test_regex_diagnosis_extraction(self):
        """Test MI diagnosis extraction using regex."""
        patient_id = "302"
        hadm_id = "adm_clin_2"

        notes = pd.DataFrame(
            {
                "patient_id": [patient_id],
                "hadm_id": [hadm_id],
                "note_type": ["Discharge Summary"],
                "charttime": [datetime(2023, 3, 5, 14, 0)],
                "text": ["Final diagnosis: Acute Myocardial Infarction (MI)."],
            }
        )

        self.mock_notes_loader.get_patient_notes.return_value = notes
        self.mock_llm_client.enabled = False

        evidence = self.extractor.collect_evidence(patient_id, hadm_id)

        self.assertEqual(len(evidence["diagnoses"]), 1)
        self.assertEqual(evidence["diagnoses"][0]["diagnosis"], "MI Diagnosis")

    def test_llm_extraction(self):
        """Test symptom and diagnosis extraction using LLM."""
        patient_id = "303"
        hadm_id = "adm_clin_3"

        notes = pd.DataFrame(
            {
                "patient_id": [patient_id],
                "hadm_id": [hadm_id],
                "note_type": ["Physician"],
                "charttime": [datetime(2023, 3, 10, 9, 0)],
                "text": ["Pt reports chest tightness. Impression is NSTEMI."],
            }
        )

        llm_output = {
            "symptoms": [
                {
                    "symptom": "Chest Pain",
                    "context": "Pt reports chest tightness.",
                    "onset_time": "2023-03-10T08:00:00",
                }
            ],
            "diagnoses": [{"diagnosis": "NSTEMI", "diagnosis_date": "2023-03-10"}],
        }

        self.mock_notes_loader.get_patient_notes.return_value = notes
        self.mock_llm_client.enabled = True
        self.mock_llm_client.extract_json.return_value = llm_output

        evidence = self.extractor.collect_evidence(patient_id, hadm_id)

        self.assertEqual(evidence["metadata"]["extraction_mode"], "llm")
        self.assertTrue(evidence["ischemic_symptoms_present"])
        self.assertEqual(len(evidence["symptoms"]), 1)
        self.assertEqual(len(evidence["diagnoses"]), 1)
        self.assertEqual(evidence["diagnoses"][0]["diagnosis"], "NSTEMI")


if __name__ == "__main__":
    unittest.main()
