# Pipeline Module
# Provides a reusable, ordered chain of image processing operations
#
# Uses only base dependencies (numpy); all operations handed to a Pipeline
# may come from any module, including those with optional backends.

from .pipeline import Pipeline, Step

__all__ = ["Pipeline", "Step"]
