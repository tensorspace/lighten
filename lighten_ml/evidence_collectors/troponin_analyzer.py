"""Troponin analyzer for detecting myocardial infarction patterns."""
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from datetime import datetime
from .base_evidence_collector import BaseEvidenceCollector

class TroponinAnalyzer(BaseEvidenceCollector):
    """Analyzes troponin levels to detect myocardial infarction patterns."""
    
    # Diagnostic threshold for Troponin T in ng/mL
    TROPONIN_THRESHOLD = 0.014
    
    def __init__(self, lab_data_loader: Any):
        """Initialize the TroponinAnalyzer with a lab data loader.
        
        Args:
            lab_data_loader: Instance of LabDataLoader for accessing lab data
        """
        super().__init__(lab_data_loader=lab_data_loader)
    
    def collect_evidence(self, patient_id: str) -> Dict[str, Any]:
        """Collect and analyze troponin evidence for a patient.
        
        Args:
            patient_id: The ID of the patient
            
        Returns:
            Dictionary containing troponin analysis results
        """
        evidence = self._get_evidence_base()
        
        # Get all troponin tests for the patient
        troponin_tests = self.lab_data_loader.get_troponin_tests(patient_id)
        
        if troponin_tests.empty:
            evidence['troponin_available'] = False
            return evidence
        
        # Process troponin values
        processed = self._process_troponin_tests(troponin_tests)
        
        # Check for MI criteria
        criteria_met, criteria_details = self._check_mi_criteria(processed['values'])
        
        evidence.update({
            'troponin_available': True,
            'troponin_tests': processed['values'],
            'max_troponin': processed['max_value'],
            'mi_criteria_met': criteria_met,
            'criteria_details': criteria_details,
            'sources': [{
                'type': 'lab',
                'description': 'Troponin test results',
                'count': len(processed['values'])
            }]
        })
        
        return evidence
    
    def _process_troponin_tests(self, troponin_tests: Any) -> Dict[str, Any]:
        """Process raw troponin test data.
        
        Args:
            troponin_tests: DataFrame containing troponin test results
            
        Returns:
            Dictionary containing processed troponin values and metadata
        """
        # Extract relevant columns and clean data
        processed = []
        max_value = 0.0
        
        for _, test in troponin_tests.iterrows():
            try:
                value = float(test['value'])
                timestamp = test.get('charttime', None)
                
                # Track maximum value
                if value > max_value:
                    max_value = value
                
                processed.append({
                    'value': value,
                    'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
                    'above_threshold': value > self.TROPONIN_THRESHOLD,
                    'test_id': test.get('itemid'),
                    'test_name': test.get('label', 'Troponin T')
                })
            except (ValueError, TypeError):
                continue
        
        # Sort by timestamp
        processed.sort(key=lambda x: x.get('timestamp', ''))
        
        return {
            'values': processed,
            'max_value': max_value,
            'threshold': self.TROPONIN_THRESHOLD
        }
    
    def _check_mi_criteria(self, troponin_values: List[Dict]) -> Tuple[bool, Dict]:
        """Check if troponin values meet MI criteria.
        
        Args:
            troponin_values: List of processed troponin values with timestamps
            
        Returns:
            Tuple of (criteria_met, criteria_details)
        """
        if not troponin_values:
            return False, {'reason': 'No troponin values available'}
        
        # Check for single value >5x threshold
        max_value = max(t['value'] for t in troponin_values)
        if max_value > 5 * self.TROPONIN_THRESHOLD:
            return True, {
                'criteria': 'Single value >5x threshold',
                'value': max_value,
                'threshold': 5 * self.TROPONIN_THRESHOLD
            }
        
        # Need at least 2 values to check for rise/fall patterns
        if len(troponin_values) < 2:
            return False, {
                'reason': 'Insufficient data points for pattern analysis',
                'required': 2,
                'available': len(troponin_values)
            }
        
        # Check for rise pattern
        rise_result = self._check_rise_pattern(troponin_values)
        if rise_result['met']:
            return True, {
                'criteria': 'Rise pattern detected',
                'details': rise_result
            }
        
        # Check for fall pattern
        fall_result = self._check_fall_pattern(troponin_values)
        if fall_result['met']:
            return True, {
                'criteria': 'Fall pattern detected',
                'details': fall_result
            }
        
        return False, {'reason': 'No MI criteria met'}
    
    def _check_rise_pattern(self, values: List[Dict]) -> Dict:
        """Check for rise pattern in troponin values.
        
        Args:
            values: List of processed troponin values
            
        Returns:
            Dictionary with rise pattern analysis
        """
        result = {
            'met': False,
            'pattern': 'rise',
            'details': []
        }
        
        for i in range(1, len(values)):
            prev = values[i-1]
            curr = values[i]
            
            # Skip if we can't compare these values
            if 'value' not in prev or 'value' not in curr:
                continue
                
            prev_val = prev['value']
            curr_val = curr['value']
            
            # Case 1: Baseline below threshold, subsequent above threshold
            if prev_val <= self.TROPONIN_THRESHOLD < curr_val:
                result['met'] = True
                result['details'].append({
                    'type': 'below_to_above_threshold',
                    'from': prev_val,
                    'to': curr_val,
                    'threshold': self.TROPONIN_THRESHOLD,
                    'indices': (i-1, i)
                })
            
            # Case 2: Significant increase from elevated baseline (≥50%)
            elif (prev_val > self.TROPONIN_THRESHOLD and 
                  curr_val >= 1.5 * prev_val):
                result['met'] = True
                result['details'].append({
                    'type': 'significant_increase',
                    'from': prev_val,
                    'to': curr_val,
                    'increase_pct': ((curr_val - prev_val) / prev_val) * 100,
                    'threshold_pct': 50,
                    'indices': (i-1, i)
                })
        
        return result
    
    def _check_fall_pattern(self, values: List[Dict]) -> Dict:
        """Check for fall pattern in troponin values.
        
        Args:
            values: List of processed troponin values
            
        Returns:
            Dictionary with fall pattern analysis
        """
        result = {
            'met': False,
            'pattern': 'fall',
            'details': []
        }
        
        for i in range(1, len(values)):
            prev = values[i-1]
            curr = values[i]
            
            # Skip if we can't compare these values
            if 'value' not in prev or 'value' not in curr:
                continue
                
            prev_val = prev['value']
            curr_val = curr['value']
            
            # Case 1: Peak above threshold with subsequent decline (≥25%)
            if (prev_val > self.TROPONIN_THRESHOLD and 
                curr_val <= 0.75 * prev_val):
                result['met'] = True
                result['details'].append({
                    'type': 'significant_decline',
                    'from': prev_val,
                    'to': curr_val,
                    'decrease_pct': ((prev_val - curr_val) / prev_val) * 100,
                    'threshold_pct': 25,
                    'indices': (i-1, i)
                })
            
            # Case 2: Declining from elevated baseline (≥25% decrease)
            elif (prev_val > self.TROPONIN_THRESHOLD and 
                  curr_val <= 0.75 * prev_val):
                result['met'] = True
                result['details'].append({
                    'type': 'declining_from_elevated',
                    'from': prev_val,
                    'to': curr_val,
                    'decrease_pct': ((prev_val - curr_val) / prev_val) * 100,
                    'threshold_pct': 25,
                    'indices': (i-1, i)
                })
        
        return result
