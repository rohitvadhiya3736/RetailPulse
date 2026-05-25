"""Custom exception hierarchy for RetailPulse."""


class RetailPulseError(Exception):
    """Base exception for all RetailPulse errors."""


class DataValidationError(RetailPulseError):
    """Raised when data fails schema or quality checks."""


class ModelTrainingError(RetailPulseError):
    """Raised when model training fails validation gates."""


class ForecastingError(RetailPulseError):
    """Raised when forecasting pipeline fails."""


class DriftDetectionError(RetailPulseError):
    """Raised when drift monitoring fails."""
