"""cim-mlir's ONNX front end.

Reads an ONNX model and emits the MLIR text the cim pipeline consumes.
Python rather than C++ on purpose -- see python/README.md for the
reasoning, which is about CI blast radius rather than convenience.
"""

from .emit import emit_module, sanitize_symbol
from .refusal import Refusal

__all__ = ["emit_module", "sanitize_symbol", "Refusal"]
