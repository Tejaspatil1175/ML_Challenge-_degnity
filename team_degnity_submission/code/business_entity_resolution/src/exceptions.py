"""Typed exceptions for the Business Entity Resolution pipeline.

Ensures strict error handling and fail-loud principles across all stages.
"""


class EntityResolutionError(Exception):
    """Base exception for all domain-specific errors in the pipeline."""
    pass


class ConfigurationError(EntityResolutionError):
    """Raised when configuration values, paths, or settings are missing/invalid."""
    pass


class DataLoadError(EntityResolutionError):
    """Raised when data loading, parsing, or file access fails."""
    pass


class SchemaError(EntityResolutionError):
    """Raised when input datasets violate required column schemas or ID formats."""
    pass


class NormalizationError(EntityResolutionError):
    """Raised when text/address normalization encounters unrecoverable errors."""
    pass


class BlockingError(EntityResolutionError):
    """Raised when candidate generation fails or generates empty candidate pools."""
    pass


class FeatureExtractionError(EntityResolutionError):
    """Raised when feature matrix generation contains missing/NaN values."""
    pass


class ModelTrainingError(EntityResolutionError):
    """Raised when classifier training or cross-validation fails."""
    pass


class EvaluationError(EntityResolutionError):
    """Raised during metric computation or threshold calibration."""
    pass


class SubmissionValidationError(EntityResolutionError):
    """Raised when submission TSV outputs violate competition formatting constraints."""
    pass
