"""Rule engine for Myocardial Infarction (MI) detection."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from .base_rule_engine import BaseRuleEngine, RuleResult

@dataclass
class MIRuleEngineConfig:
    """Configuration for the MI Rule Engine."""
    # Troponin threshold in ng/mL (Troponin T)
    troponin_threshold: float = 0.014
    
    # Threshold for considering a single high troponin value (5x upper limit of normal)
    single_value_threshold: float = 0.07  # 5 * 0.014
    
    # Minimum percentage increase for significant rise (50%)
    min_percent_increase: float = 50.0
    
    # Minimum percentage decrease for significant fall (25%)
    min_percent_decrease: float = 25.0
    
    # Minimum number of troponin measurements required for trend analysis
    min_troponin_measurements: int = 2
    
    # Time window in hours to look for dynamic changes
    dynamic_change_window_hours: int = 72
    
    # Required number of criteria from group B (ischemia evidence)
    required_ischemia_criteria: int = 1
    
    # Whether to require both criteria A and B for MI diagnosis
    require_both_criteria: bool = True
    
    # Whether to consider clinical symptoms as evidence
    consider_clinical_symptoms: bool = True
    
    # Whether to consider ECG findings as evidence
    consider_ecg_findings: bool = True
    
    # Whether to consider imaging findings as evidence
    consider_imaging_findings: bool = True
    
    # Whether to consider angiographic findings as evidence
    consider_angiographic_findings: bool = True
    
    # Confidence thresholds
    confidence_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            'high': 0.9,
            'medium': 0.7,
            'low': 0.5
        }
    )
    
    # Default confidence level when evidence is missing
    default_confidence: float = 0.5

class MIRuleEngine(BaseRuleEngine[MIRuleEngineConfig]):
    """Rule engine for detecting Myocardial Infarction based on clinical evidence."""
    
    def __init__(self, config: Optional[MIRuleEngineConfig] = None):
        """Initialize the MI Rule Engine.
        
        Args:
            config: Configuration for the rule engine. If None, defaults will be used.
        """
        super().__init__(config or MIRuleEngineConfig())
    
    def evaluate(self, evidence: Dict[str, Any]) -> RuleResult:
        """Evaluate evidence for Myocardial Infarction.
        
        Args:
            evidence: Dictionary containing evidence from data sources
            
        Returns:
            RuleResult with the evaluation
        """
        # Initialize result components
        criteria_met = {
            'A': False,  # Biomarker criteria
            'B': False   # Ischemia criteria
        }
        
        confidence = {
            'A': 0.0,
            'B': 0.0
        }
        
        evidence_items = []
        details = {
            'criteria_A': {},
            'criteria_B': {}
        }
        
        # Evaluate Criteria A: Biomarker evidence
        a_result = self._evaluate_criteria_a(evidence.get('troponin', {}))
        criteria_met['A'] = a_result['met']
        confidence['A'] = a_result['confidence']
        details['criteria_A'] = a_result['details']
        evidence_items.extend(a_result.get('evidence', []))
        
        # Evaluate Criteria B: Ischemia evidence
        b_result = self._evaluate_criteria_b(evidence)
        criteria_met['B'] = b_result['met']
        confidence['B'] = b_result['confidence']
        details['criteria_B'] = b_result['details']
        evidence_items.extend(b_result.get('evidence', []))
        
        # Determine overall result
        if self.config.require_both_criteria:
            passed = criteria_met['A'] and criteria_met['B']
            overall_confidence = min(confidence['A'], confidence['B'])
        else:
            passed = criteria_met['A'] or criteria_met['B']
            overall_confidence = max(confidence['A'], confidence['B'])
        
        # Add summary to details
        details['summary'] = {
            'criteria_met': criteria_met,
            'confidence_scores': confidence,
            'overall_confidence': overall_confidence,
            'decision_threshold': self.config.confidence_thresholds['medium']
        }
        
        return self._create_result(
            passed=passed,
            confidence=overall_confidence,
            evidence=evidence_items,
            details=details
        )
    
    def _evaluate_criteria_a(self, troponin_evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate Criteria A: Biomarker evidence.
        
        Args:
            troponin_evidence: Dictionary containing troponin test results
            
        Returns:
            Dictionary with evaluation results
        """
        result = {
            'met': False,
            'confidence': 0.0,
            'details': {},
            'evidence': []
        }
        
        if not troponin_evidence or not troponin_evidence.get('troponin_available', False):
            result['details'] = {'reason': 'No troponin data available'}
            return result
        
        troponin_tests = troponin_evidence.get('troponin_tests', [])
        max_troponin = troponin_evidence.get('max_troponin', 0)
        
        # Check for single high value (>5x threshold)
        if max_troponin >= self.config.single_value_threshold:
            result.update({
                'met': True,
                'confidence': 0.95,  # High confidence for clear positive
                'details': {
                    'criteria_met': 'Single high value',
                    'value': max_troponin,
                    'threshold': self.config.single_value_threshold
                },
                'evidence': [{
                    'type': 'troponin',
                    'description': f'Single high troponin value: {max_troponin:.3f} ng/mL',
                    'significance': 'Meets MI criteria with single high value',
                    'confidence': 0.95
                }]
            })
            return result
        
        # Check for dynamic pattern if enough measurements
        if len(troponin_tests) >= self.config.min_troponin_measurements:
            # Check for rise pattern
            rise_pattern = self._check_rise_pattern(troponin_tests)
            if rise_pattern['met']:
                result.update({
                    'met': True,
                    'confidence': 0.85,  # High confidence for clear pattern
                    'details': {
                        'criteria_met': 'Rise pattern',
                        'pattern_details': rise_pattern['details']
                    },
                    'evidence': [{
                        'type': 'troponin',
                        'description': 'Rise pattern detected in troponin values',
                        'significance': 'Meets MI criteria with rise pattern',
                        'confidence': 0.85,
                        'details': rise_pattern['details']
                    }]
                })
                return result
            
            # Check for fall pattern
            fall_pattern = self._check_fall_pattern(troponin_tests)
            if fall_pattern['met']:
                result.update({
                    'met': True,
                    'confidence': 0.85,  # High confidence for clear pattern
                    'details': {
                        'criteria_met': 'Fall pattern',
                        'pattern_details': fall_pattern['details']
                    },
                    'evidence': [{
                        'type': 'troponin',
                        'description': 'Fall pattern detected in troponin values',
                        'significance': 'Meets MI criteria with fall pattern',
                        'confidence': 0.85,
                        'details': fall_pattern['details']
                    }]
                })
                return result
        
        # If we get here, no criteria were met
        result.update({
            'met': False,
            'confidence': 0.7 if max_troponin > 0 else 0.5,
            'details': {
                'reason': 'No troponin criteria met',
                'max_troponin': max_troponin,
                'threshold': self.config.troponin_threshold,
                'single_value_threshold': self.config.single_value_threshold
            }
        })
        
        return result
    
    def _evaluate_criteria_b(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate Criteria B: Ischemia evidence.
        
        Args:
            evidence: Dictionary containing all evidence
            
        Returns:
            Dictionary with evaluation results
        """
        result = {
            'met': False,
            'confidence': 0.0,
            'details': {},
            'evidence': []
        }
        
        # Check each type of ischemia evidence
        evidence_sources = []
        
        # 1. Symptoms
        if self.config.consider_clinical_symptoms:
            symptoms = evidence.get('symptoms', [])
            if symptoms:
                evidence_sources.append({
                    'type': 'symptoms',
                    'count': len(symptoms),
                    'confidence': 0.7,  # Moderate confidence for symptoms alone
                    'details': symptoms
                })
        
        # 2. ECG findings
        if self.config.consider_ecg_findings:
            ecg_evidence = evidence.get('ecg', {})
            ecg_findings = ecg_evidence.get('ecg_findings', [])
            mi_related_ecg = [f for f in ecg_findings if f.get('mi_related', False)]
            
            if mi_related_ecg:
                evidence_sources.append({
                    'type': 'ecg',
                    'count': len(mi_related_ecg),
                    'confidence': 0.9,  # High confidence for ECG findings
                    'details': mi_related_ecg
                })
        
        # 3. Imaging findings (placeholder - would be implemented similarly)
        if self.config.consider_imaging_findings:
            imaging = evidence.get('imaging', {})
            if imaging.get('wall_motion_abnormalities', False):
                evidence_sources.append({
                    'type': 'imaging',
                    'confidence': 0.85,
                    'details': imaging
                })
        
        # 4. Angiographic findings (placeholder - would be implemented similarly)
        if self.config.consider_angiographic_findings:
            angio = evidence.get('angiography', {})
            if angio.get('thrombus', False) or angio.get('occlusion', False):
                evidence_sources.append({
                    'type': 'angiography',
                    'confidence': 0.95,  # Very high confidence for direct visualization
                    'details': angio
                })
        
        # Determine if criteria are met
        met = len(evidence_sources) >= self.config.required_ischemia_criteria
        
        # Calculate confidence based on evidence sources
        if evidence_sources:
            # Weighted average of confidence from all sources
            total_weight = sum(src.get('confidence', 0) for src in evidence_sources)
            avg_confidence = total_weight / len(evidence_sources)
        else:
            avg_confidence = 0.0
        
        # Update result
        result.update({
            'met': met,
            'confidence': avg_confidence,
            'details': {
                'evidence_sources': [{
                    'type': src['type'],
                    'confidence': src['confidence'],
                    'count': src.get('count', 1)
                } for src in evidence_sources],
                'required_sources': self.config.required_ischemia_criteria,
                'found_sources': len(evidence_sources)
            },
            'evidence': [{
                'type': 'ischemia_evidence',
                'description': f'Found {len(evidence_sources)} source(s) of ischemia evidence',
                'significance': 'Meets ischemia criteria' if met else 'Insufficient ischemia evidence',
                'confidence': avg_confidence,
                'details': {
                    'sources': [src['type'] for src in evidence_sources],
                    'required': self.config.required_ischemia_criteria
                }
            }]
        })
        
        return result
    
    def _check_rise_pattern(self, troponin_tests: List[Dict]) -> Dict[str, Any]:
        """Check for a rise pattern in troponin values.
        
        Args:
            troponin_tests: List of troponin test results
            
        Returns:
            Dictionary with pattern detection results
        """
        result = {
            'met': False,
            'details': []
        }
        
        for i in range(1, len(troponin_tests)):
            prev = troponin_tests[i-1]
            curr = troponin_tests[i]
            
            # Skip if we don't have valid values or timestamps
            if 'value' not in prev or 'value' not in curr or 'timestamp' not in prev or 'timestamp' not in curr:
                continue
                
            prev_val = prev['value']
            curr_val = curr['value']
            
            # Calculate time difference in hours
            try:
                prev_time = datetime.fromisoformat(prev['timestamp'].replace('Z', '+00:00'))
                curr_time = datetime.fromisoformat(curr['timestamp'].replace('Z', '+00:00'))
                hours_diff = (curr_time - prev_time).total_seconds() / 3600
                
                # Skip if measurements are too far apart
                if hours_diff > self.config.dynamic_change_window_hours:
                    continue
                
                # Case 1: Below threshold to above threshold
                if prev_val <= self.config.troponin_threshold < curr_val:
                    result['met'] = True
                    result['details'].append({
                        'type': 'below_to_above_threshold',
                        'from': prev_val,
                        'to': curr_val,
                        'threshold': self.config.troponin_threshold,
                        'hours_apart': hours_diff,
                        'indices': (i-1, i)
                    })
                
                # Case 2: Significant increase from elevated baseline (≥50%)
                elif (prev_val > self.config.troponin_threshold and 
                      curr_val >= (1 + self.config.min_percent_increase/100) * prev_val):
                    percent_increase = ((curr_val - prev_val) / prev_val) * 100
                    result['met'] = True
                    result['details'].append({
                        'type': 'significant_increase',
                        'from': prev_val,
                        'to': curr_val,
                        'percent_increase': percent_increase,
                        'required_percent': self.config.min_percent_increase,
                        'hours_apart': hours_diff,
                        'indices': (i-1, i)
                    })
                    
            except (ValueError, TypeError):
                continue
        
        return result
    
    def _check_fall_pattern(self, troponin_tests: List[Dict]) -> Dict[str, Any]:
        """Check for a fall pattern in troponin values.
        
        Args:
            troponin_tests: List of troponin test results
            
        Returns:
            Dictionary with pattern detection results
        """
        result = {
            'met': False,
            'details': []
        }
        
        for i in range(1, len(troponin_tests)):
            prev = troponin_tests[i-1]
            curr = troponin_tests[i]
            
            # Skip if we don't have valid values or timestamps
            if 'value' not in prev or 'value' not in curr or 'timestamp' not in prev or 'timestamp' not in curr:
                continue
                
            prev_val = prev['value']
            curr_val = curr['value']
            
            # Calculate time difference in hours
            try:
                prev_time = datetime.fromisoformat(prev['timestamp'].replace('Z', '+00:00'))
                curr_time = datetime.fromisoformat(curr['timestamp'].replace('Z', '+00:00'))
                hours_diff = (curr_time - prev_time).total_seconds() / 3600
                
                # Skip if measurements are too far apart
                if hours_diff > self.config.dynamic_change_window_hours:
                    continue
                
                # Check for significant decline (≥25%)
                if (prev_val > self.config.troponin_threshold and 
                    curr_val <= (1 - self.config.min_percent_decrease/100) * prev_val):
                    percent_decrease = ((prev_val - curr_val) / prev_val) * 100
                    result['met'] = True
                    result['details'].append({
                        'type': 'significant_decline',
                        'from': prev_val,
                        'to': curr_val,
                        'percent_decrease': percent_decrease,
                        'required_percent': self.config.min_percent_decrease,
                        'hours_apart': hours_diff,
                        'indices': (i-1, i)
                    })
                    
            except (ValueError, TypeError):
                continue
        
        return result
