"""Domain-specific checkpoint adapters."""

from spot_checkpoint.adapters.numpy_dict import NumpyDictAdapter
from spot_checkpoint.adapters.scipy_opt import (
    ScipyOptimizeAdapter,
    ScipySparseLinalgAdapter,
)

__all__ = [
    "NumpyDictAdapter",
    "ScipyOptimizeAdapter",
    "ScipySparseLinalgAdapter",
]
