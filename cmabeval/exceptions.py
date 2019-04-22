"""
Custom exceptions.

"""


class CMABEvalException(Exception):
    """Base class for all custom exceptions."""
    pass


class InsufficientData(CMABEvalException):
    """Raise when model cannot be fit due to insufficient data."""
    pass


class NotFitted(CMABEvalException):
    """Raise when a model has not been fit and a method
    is being called that depends on it having been fit.
    """
    pass
