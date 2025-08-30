"""
Configuration module for evidence extraction methods.

This module provides configuration classes and utilities for managing
different extraction approaches (regex vs LLM-based) in the clinical pipeline.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from .hybrid_evidence_extractor import ExtractionMethod


@dataclass
class ExtractionConfig:
    """
    Configuration for evidence extraction methods.
    """

    # Primary extraction method
    clinical_method: ExtractionMethod = ExtractionMethod.HYBRID
    ecg_method: ExtractionMethod = ExtractionMethod.HYBRID

    # Confidence thresholds
    clinical_confidence_threshold: float = 0.7
    ecg_confidence_threshold: float = 0.7

    # Processing limits
    max_notes_per_admission: Optional[int] = None

    # LLM-specific settings
    llm_temperature: float = 0.1  # Low temperature for consistent extraction
    llm_max_tokens: int = 2000
    llm_timeout_seconds: int = 30

    # Fallback behavior
    enable_regex_fallback: bool = True
    log_extraction_details: bool = True

    # Performance settings
    enable_parallel_processing: bool = False
    batch_size: int = 5

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "clinical_method": self.clinical_method.value,
            "ecg_method": self.ecg_method.value,
            "clinical_confidence_threshold": self.clinical_confidence_threshold,
            "ecg_confidence_threshold": self.ecg_confidence_threshold,
            "max_notes_per_admission": self.max_notes_per_admission,
            "llm_temperature": self.llm_temperature,
            "llm_max_tokens": self.llm_max_tokens,
            "llm_timeout_seconds": self.llm_timeout_seconds,
            "enable_regex_fallback": self.enable_regex_fallback,
            "log_extraction_details": self.log_extraction_details,
            "enable_parallel_processing": self.enable_parallel_processing,
            "batch_size": self.batch_size,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ExtractionConfig":
        """Create configuration from dictionary."""
        config = cls()

        # Handle enum conversion
        if "clinical_method" in config_dict:
            config.clinical_method = ExtractionMethod(config_dict["clinical_method"])
        if "ecg_method" in config_dict:
            config.ecg_method = ExtractionMethod(config_dict["ecg_method"])

        # Set other attributes
        for key, value in config_dict.items():
            if key not in ["clinical_method", "ecg_method"] and hasattr(config, key):
                setattr(config, key, value)

        return config

    def validate(self) -> bool:
        """Validate configuration parameters."""
        if not 0.0 <= self.clinical_confidence_threshold <= 1.0:
            raise ValueError(
                "Clinical confidence threshold must be between 0.0 and 1.0"
            )

        if not 0.0 <= self.ecg_confidence_threshold <= 1.0:
            raise ValueError("ECG confidence threshold must be between 0.0 and 1.0")

        if (
            self.max_notes_per_admission is not None
            and self.max_notes_per_admission < 1
        ):
            raise ValueError("Max notes per admission must be positive")

        if not 0.0 <= self.llm_temperature <= 2.0:
            raise ValueError("LLM temperature must be between 0.0 and 2.0")

        if self.llm_max_tokens < 100:
            raise ValueError("LLM max tokens must be at least 100")

        if self.llm_timeout_seconds < 5:
            raise ValueError("LLM timeout must be at least 5 seconds")

        if self.batch_size < 1:
            raise ValueError("Batch size must be positive")

        return True


# Predefined configurations for different use cases
FAST_CONFIG = ExtractionConfig(
    clinical_method=ExtractionMethod.REGEX,
    ecg_method=ExtractionMethod.REGEX,
    max_notes_per_admission=5,
    enable_parallel_processing=True,
    log_extraction_details=False,
)

ACCURATE_CONFIG = ExtractionConfig(
    clinical_method=ExtractionMethod.LLM,
    ecg_method=ExtractionMethod.LLM,
    clinical_confidence_threshold=0.8,
    ecg_confidence_threshold=0.8,
    llm_temperature=0.05,
    max_notes_per_admission=10,
    log_extraction_details=True,
)

BALANCED_CONFIG = ExtractionConfig(
    clinical_method=ExtractionMethod.HYBRID,
    ecg_method=ExtractionMethod.HYBRID,
    clinical_confidence_threshold=0.7,
    ecg_confidence_threshold=0.7,
    enable_regex_fallback=True,
    max_notes_per_admission=8,
    log_extraction_details=True,
)

RESEARCH_CONFIG = ExtractionConfig(
    clinical_method=ExtractionMethod.LLM,
    ecg_method=ExtractionMethod.LLM,
    clinical_confidence_threshold=0.6,
    ecg_confidence_threshold=0.6,
    llm_temperature=0.1,
    max_notes_per_admission=None,  # Process all notes
    enable_parallel_processing=False,
    log_extraction_details=True,
)
