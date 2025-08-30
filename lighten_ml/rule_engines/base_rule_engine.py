"""Base class for rule engines."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Generic, List, Optional, TypeVar

# Define a generic type variable for configuration
T = TypeVar("T")


@dataclass
class RuleResult:
    """Container for rule evaluation results."""

    passed: bool
    confidence: float
    evidence: List[Dict[str, Any]]
    details: Dict[str, Any]
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary."""
        return {
            "passed": self.passed,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "details": self.details,
            "timestamp": self.timestamp,
        }


class BaseRuleEngine(ABC, Generic[T]):
    """Abstract base class for all rule engines."""

    def __init__(self, config: Optional[T] = None):
        """Initialize the rule engine with optional configuration.

        Args:
            config: Configuration object for the rule engine
        """
        self.config = config

    @abstractmethod
    def evaluate(self, evidence: Dict[str, Any]) -> RuleResult:
        """Evaluate the evidence against the rules.

        Args:
            evidence: Dictionary containing evidence to evaluate

        Returns:
            RuleResult containing the evaluation results
        """
        pass

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format.

        Returns:
            ISO formatted current timestamp
        """
        return datetime.utcnow().isoformat()

    def _create_result(
        self,
        passed: bool,
        confidence: float,
        evidence: List[Dict[str, Any]],
        details: Dict[str, Any],
    ) -> RuleResult:
        """Create a RuleResult object.

        Args:
            passed: Whether the rule passed
            confidence: Confidence score (0.0 to 1.0)
            evidence: List of evidence items
            details: Additional details about the evaluation

        Returns:
            RuleResult object
        """
        return RuleResult(
            passed=passed,
            confidence=max(0.0, min(1.0, confidence)),  # Clamp to [0, 1]
            evidence=evidence,
            details=details,
            timestamp=self._get_timestamp(),
        )
