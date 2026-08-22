class MLWeaveError(Exception):
    """Base exception shared across the mlweave package."""


class MLWeaveConfigurationError(MLWeaveError):
    """Raised when mlweave configuration is invalid."""


class MLWeaveValidationError(MLWeaveError):
    """Raised when an mlweave validation fails."""
