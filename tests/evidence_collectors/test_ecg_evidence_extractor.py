import unittest
import pandas as pd
from unittest.mock import MagicMock, patch
from datetime import datetime

from lighten_ml.evidence_collectors.ecg_evidence_extractor import ECGEvidenceExtractor

class TestECGEvidenceExtractor(unittest.TestCase):

    def setUp(self):
        """Set up mock data loader and LLM client for tests."""
        self.mock_notes_loader = MagicMock()
        self.mock_llm_client = MagicMock()
        self.extractor = ECGEvidenceExtractor(self.mock_notes_loader, llm_client=self.mock_llm_client)

    def test_regex_extraction(self):
        """Test ECG evidence extraction using regex patterns."""
        patient_id = '201'
        hadm_id = 'adm_ecg_1'
        
        ecg_notes = pd.DataFrame({
            'patient_id': [patient_id],
            'hadm_id': [hadm_id],
            'note_type': ['ECG'],
            'charttime': [datetime(2023, 2, 1, 10, 0)],
            'text': ['ECG shows new ST segment elevation in leads V2, V3. Also T wave inversion.']
        })
        
        self.mock_notes_loader.get_patient_notes.return_value = ecg_notes
        self.mock_llm_client.enabled = False

        evidence = self.extractor.collect_evidence(patient_id, hadm_id)
        
        self.assertEqual(evidence['metadata']['extraction_mode'], 'regex')
        self.assertEqual(len(evidence['ecg_findings']), 2)
        self.assertTrue(any(f['finding'] == 'ST Elevation' for f in evidence['ecg_findings']))
        self.assertTrue(any(f['finding'] == 'T Wave Inversion' for f in evidence['ecg_findings']))

    @patch('lighten_ml.evidence_collectors.ecg_evidence_extractor.ECGEvidenceExtractor._post_process_llm_findings')
    def test_llm_extraction_path(self, mock_post_process):
        """Test that the LLM extraction path is taken when enabled."""
        patient_id = '202'
        hadm_id = 'adm_ecg_2'
        
        ecg_notes = pd.DataFrame({
            'patient_id': [patient_id],
            'hadm_id': [hadm_id],
            'note_type': ['ECG'],
            'charttime': [datetime(2023, 2, 2, 11, 0)],
            'text': ['1.5mm ST elevation in V4, V5.']
        })
        
        llm_output = [{'finding': 'ST Elevation'}] # Dummy output
        mock_post_process.return_value = llm_output
        self.mock_notes_loader.get_patient_notes.return_value = ecg_notes
        self.mock_llm_client.enabled = True
        self.mock_llm_client.extract_json.return_value = {'ecg_findings': llm_output}

        evidence = self.extractor.collect_evidence(patient_id, hadm_id)

        self.assertEqual(evidence['metadata']['extraction_mode'], 'llm_validated')
        self.assertEqual(len(evidence['ecg_findings']), 1)
        mock_post_process.assert_called_once()

    def test_llm_post_validation_contiguous_leads(self):
        """Test validation of contiguous leads from LLM output."""
        # V2 and V3 are contiguous (anterior)
        finding_valid = {
            'finding': 'ST Elevation', 'is_new': True, 'leads': ['V2', 'V3'],
            'measurements': {'st_elevation_mm': 1.5}
        }
        # V1 and V6 are not contiguous
        finding_invalid = {
            'finding': 'ST Elevation', 'is_new': True, 'leads': ['V1', 'V6'],
            'measurements': {'st_elevation_mm': 1.5}
        }

        processed = self.extractor._post_process_llm_findings([finding_valid, finding_invalid])
        
        self.assertEqual(len(processed), 1)
        self.assertEqual(processed[0]['finding'], 'ST Elevation')

    def test_llm_post_validation_thresholds(self):
        """Test validation of measurement thresholds from LLM output."""
        # Meets ST elevation threshold of >= 1mm
        finding_valid = {
            'finding': 'ST Elevation', 'is_new': True, 'leads': ['II', 'III'],
            'measurements': {'st_elevation_mm': 1.2}
        }
        # Below ST depression threshold of >= 0.5mm
        finding_invalid = {
            'finding': 'ST Depression', 'is_new': True, 'leads': ['V5', 'V6'],
            'measurements': {'st_depression_mm': 0.3}
        }

        processed = self.extractor._post_process_llm_findings([finding_valid, finding_invalid])
        
        self.assertEqual(len(processed), 1)
        self.assertEqual(processed[0]['finding'], 'ST Elevation')

    def test_llm_post_validation_is_new_flag(self):
        """Test validation of the 'is_new' flag from LLM output."""
        finding_valid = {'finding': 'T Wave Inversion', 'is_new': True, 'leads': ['V1', 'V2'], 'measurements': {'t_wave_inversion_mm': 2.0}}
        finding_invalid = {'finding': 'T Wave Inversion', 'is_new': False, 'leads': ['V1', 'V2'], 'measurements': {'t_wave_inversion_mm': 2.0}}

        processed = self.extractor._post_process_llm_findings([finding_valid, finding_invalid])
        
        self.assertEqual(len(processed), 1)
        self.assertTrue(processed[0]['is_new'])

if __name__ == '__main__':
    unittest.main()
