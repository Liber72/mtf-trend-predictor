"""Application-specific exceptions."""


class AppError(Exception):
    """Base class for domain and infrastructure errors."""


class ConfigurationError(AppError):
    """Raised when runtime configuration is invalid."""


class DataError(AppError):
    """Raised when market data cannot be loaded or normalized."""


class ModelError(AppError):
    """Raised when training or inference fails."""


class TradingError(AppError):
    """Raised when trade execution or broker interaction fails."""


class DatabaseError(AppError):
    """Raised when persistence fails."""
