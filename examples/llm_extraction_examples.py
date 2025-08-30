"""
Usage examples for LLM-based clinical text extraction.

This module demonstrates how to use the new LLM-based extraction methods
for better accuracy and flexibility in clinical evidence collection.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from lighten_ml.data_loaders.clinical_notes_loader import ClinicalNotesLoader
from lighten_ml.evidence_collectors.extraction_config import (
    ACCURATE_CONFIG,
    BALANCED_CONFIG,
    FAST_CONFIG,
    ExtractionConfig,
)
from lighten_ml.evidence_collectors.hybrid_evidence_extractor import (
    ExtractionMethod,
    HybridEvidenceExtractor,
)
from lighten_ml.evidence_collectors.llm_clinical_evidence_extractor import (
    LLMClinicalEvidenceExtractor,
)
from lighten_ml.evidence_collectors.llm_ecg_evidence_extractor import (
    LLMECGEvidenceExtractor,
)
from lighten_ml.llm_client import LightenLLMClient


def example_1_basic_llm_clinical_extraction():
    """
    Example 1: Basic LLM-based clinical evidence extraction.

    This example shows how to use the LLM clinical extractor directly
    for high-accuracy symptom detection with context understanding.
    """
    print("=== Example 1: Basic LLM Clinical Extraction ===")

    # Initialize components
    notes_loader = ClinicalNotesLoader("path/to/clinical_notes.csv")
    llm_client = LightenLLMClient()  # Uses API key from environment

    # Create LLM-based clinical extractor
    llm_extractor = LLMClinicalEvidenceExtractor(
        notes_data_loader=notes_loader, llm_client=llm_client, max_notes=5
    )

    # Extract evidence for a patient
    patient_id = "12345"
    hadm_id = "67890"

    evidence = llm_extractor.collect_evidence(patient_id, hadm_id)

    print(f"Extraction method: {evidence['extraction_method']}")
    print(f"Symptoms found: {len(evidence['symptoms'])}")
    print(f"Negated symptoms: {evidence['negated_symptoms']}")

    # Display detailed symptom information
    for symptom in evidence["symptoms"]:
        print(f"\nSymptom: {symptom['name']}")
        print(f"  Description: {symptom['description']}")
        print(f"  Severity: {symptom['severity']}")
        print(f"  Quality: {symptom['quality']}")
        print(f"  Confidence: {symptom['confidence']:.2f}")
        if symptom["radiation"]:
            print(f"  Radiation: {', '.join(symptom['radiation'])}")


def example_2_llm_ecg_extraction():
    """
    Example 2: LLM-based ECG evidence extraction.

    This example demonstrates advanced ECG interpretation using LLM
    for better detection of ischemic changes and pathological findings.
    """
    print("\n=== Example 2: LLM ECG Extraction ===")

    # Initialize components
    notes_loader = ClinicalNotesLoader("path/to/clinical_notes.csv")
    llm_client = LightenLLMClient()

    # Create LLM-based ECG extractor
    ecg_extractor = LLMECGEvidenceExtractor(
        notes_data_loader=notes_loader, llm_client=llm_client, max_notes=3
    )

    # Extract ECG evidence
    patient_id = "12345"
    hadm_id = "67890"

    evidence = ecg_extractor.collect_evidence(patient_id, hadm_id)

    print(f"ECG findings: {len(evidence['ecg_findings'])}")
    print(f"Extraction method: {evidence['extraction_method']}")

    # Display ECG findings
    for finding in evidence["ecg_findings"]:
        print(f"\nFinding: {finding['finding']}")
        print(f"  Description: {finding['description']}")
        print(f"  Leads: {', '.join(finding['leads'])}")
        print(f"  Region: {finding['anatomical_region']}")
        print(f"  New finding: {finding['is_new']}")
        print(f"  MI-related: {finding['mi_related']}")
        print(f"  Confidence: {finding['confidence']:.2f}")
        print(f"  Clinical significance: {finding['clinical_significance']}")


def example_3_hybrid_extraction():
    """
    Example 3: Hybrid extraction with automatic fallback.

    This example shows how to use the hybrid extractor that automatically
    chooses between LLM and regex based on availability and confidence.
    """
    print("\n=== Example 3: Hybrid Extraction ===")

    # Initialize components
    notes_loader = ClinicalNotesLoader("path/to/clinical_notes.csv")
    llm_client = LightenLLMClient()

    # Create hybrid extractor
    hybrid_extractor = HybridEvidenceExtractor(
        notes_data_loader=notes_loader,
        llm_client=llm_client,
        max_notes=5,
        default_method=ExtractionMethod.HYBRID,
        confidence_threshold=0.7,
    )

    patient_id = "12345"
    hadm_id = "67890"

    # Extract clinical evidence using hybrid approach
    clinical_evidence = hybrid_extractor.collect_clinical_evidence(patient_id, hadm_id)
    print(f"Clinical extraction method used: {clinical_evidence['extraction_method']}")
    print(f"Clinical symptoms found: {len(clinical_evidence['symptoms'])}")

    # Extract ECG evidence using hybrid approach
    ecg_evidence = hybrid_extractor.collect_ecg_evidence(patient_id, hadm_id)
    print(f"ECG extraction method used: {ecg_evidence['extraction_method']}")
    print(f"ECG findings: {len(ecg_evidence['ecg_findings'])}")

    # Show available methods
    available_methods = hybrid_extractor.get_available_methods()
    print(f"Available extraction methods: {[m.value for m in available_methods]}")


def example_4_method_comparison():
    """
    Example 4: Compare different extraction methods.

    This example demonstrates how to compare results from different
    extraction methods to evaluate accuracy and coverage.
    """
    print("\n=== Example 4: Method Comparison ===")

    # Initialize components
    notes_loader = ClinicalNotesLoader("path/to/clinical_notes.csv")
    llm_client = LightenLLMClient()

    hybrid_extractor = HybridEvidenceExtractor(
        notes_data_loader=notes_loader, llm_client=llm_client, max_notes=5
    )

    patient_id = "12345"
    hadm_id = "67890"

    # Compare different methods
    methods = [ExtractionMethod.REGEX, ExtractionMethod.LLM, ExtractionMethod.HYBRID]
    results = {}

    for method in methods:
        if method in hybrid_extractor.get_available_methods():
            print(f"\n--- Testing {method.value} method ---")

            clinical_result = hybrid_extractor.collect_clinical_evidence(
                patient_id, hadm_id, method=method
            )

            results[method.value] = {
                "symptoms_count": len(clinical_result["symptoms"]),
                "extraction_method": clinical_result["extraction_method"],
                "confidence_scores": clinical_result.get("confidence_scores", {}),
            }

            print(f"Symptoms found: {results[method.value]['symptoms_count']}")
            print(f"Actual method used: {results[method.value]['extraction_method']}")

    # Summary comparison
    print("\n--- Comparison Summary ---")
    for method, result in results.items():
        avg_confidence = 0
        if result["confidence_scores"]:
            avg_confidence = sum(result["confidence_scores"].values()) / len(
                result["confidence_scores"]
            )

        print(
            f"{method}: {result['symptoms_count']} symptoms, "
            f"avg confidence: {avg_confidence:.2f}"
        )


def example_5_configuration_usage():
    """
    Example 5: Using predefined configurations.

    This example shows how to use predefined extraction configurations
    for different use cases (fast, accurate, balanced, research).
    """
    print("\n=== Example 5: Configuration Usage ===")

    # Initialize components
    notes_loader = ClinicalNotesLoader("path/to/clinical_notes.csv")
    llm_client = LightenLLMClient()

    # Test different configurations
    configs = {
        "Fast": FAST_CONFIG,
        "Accurate": ACCURATE_CONFIG,
        "Balanced": BALANCED_CONFIG,
    }

    patient_id = "12345"
    hadm_id = "67890"

    for config_name, config in configs.items():
        print(f"\n--- {config_name} Configuration ---")
        print(f"Clinical method: {config.clinical_method.value}")
        print(f"ECG method: {config.ecg_method.value}")
        print(f"Confidence threshold: {config.clinical_confidence_threshold}")
        print(f"Max notes: {config.max_notes_per_admission}")

        # Create extractor with specific configuration
        extractor = HybridEvidenceExtractor(
            notes_data_loader=notes_loader,
            llm_client=llm_client,
            max_notes=config.max_notes_per_admission,
            default_method=config.clinical_method,
            confidence_threshold=config.clinical_confidence_threshold,
        )

        # Extract evidence
        evidence = extractor.collect_clinical_evidence(patient_id, hadm_id)
        print(f"Results: {len(evidence['symptoms'])} symptoms found")


def example_6_error_handling_and_fallbacks():
    """
    Example 6: Error handling and fallback mechanisms.

    This example demonstrates how the system handles errors and
    automatically falls back to regex when LLM extraction fails.
    """
    print("\n=== Example 6: Error Handling and Fallbacks ===")

    # Initialize with potentially problematic setup
    notes_loader = ClinicalNotesLoader("path/to/clinical_notes.csv")

    # Test with no LLM client (should fallback to regex)
    print("Testing without LLM client:")
    hybrid_extractor_no_llm = HybridEvidenceExtractor(
        notes_data_loader=notes_loader,
        llm_client=None,  # No LLM available
        default_method=ExtractionMethod.LLM,  # Will fallback to regex
    )

    available_methods = hybrid_extractor_no_llm.get_available_methods()
    print(f"Available methods without LLM: {[m.value for m in available_methods]}")

    # Test with LLM client but request LLM method
    print("\nTesting LLM extraction with fallback:")
    llm_client = LightenLLMClient()
    hybrid_extractor = HybridEvidenceExtractor(
        notes_data_loader=notes_loader,
        llm_client=llm_client,
        default_method=ExtractionMethod.HYBRID,
        confidence_threshold=0.9,  # High threshold to trigger fallbacks
    )

    patient_id = "12345"
    hadm_id = "67890"

    evidence = hybrid_extractor.collect_clinical_evidence(patient_id, hadm_id)
    print(f"Final extraction method used: {evidence['extraction_method']}")
    print(f"This shows how the system adapts to different conditions")


if __name__ == "__main__":
    """
    Run all examples to demonstrate LLM-based extraction capabilities.

    Note: Make sure to set your LLM API key in environment variables:
    export TOGETHER_API_KEY="your_api_key_here"
    or
    export LLM_API_KEY="your_api_key_here"
    """

    print("LLM-Based Clinical Text Extraction Examples")
    print("=" * 50)

    # Check if API key is available
    if not (os.getenv("TOGETHER_API_KEY") or os.getenv("LLM_API_KEY")):
        print("WARNING: No LLM API key found in environment variables.")
        print("Some examples may fall back to regex extraction.")
        print("Set TOGETHER_API_KEY or LLM_API_KEY to use LLM features.\n")

    try:
        example_1_basic_llm_clinical_extraction()
        example_2_llm_ecg_extraction()
        example_3_hybrid_extraction()
        example_4_method_comparison()
        example_5_configuration_usage()
        example_6_error_handling_and_fallbacks()

        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("\nKey Benefits of LLM-based Extraction:")
        print("✅ Context-aware symptom detection")
        print("✅ Negation and temporal understanding")
        print("✅ Medical abbreviation recognition")
        print("✅ Confidence scoring for quality assessment")
        print("✅ Structured output with detailed attributes")
        print("✅ Automatic fallback to regex when needed")

    except Exception as e:
        print(f"Error running examples: {str(e)}")
        print("Make sure all dependencies are installed and API keys are set.")
