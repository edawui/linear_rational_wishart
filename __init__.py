"""Wishart Processes Module"""

__version__ = "0.1.0"

# Available modules
__all__ = [
    "calibration",
    "components",
    "config",
    "core",
    "curves",
    "data",
    "enums",
    "examples",
    "math",
    "models",
    "others",
    "pricing",
    "saved",
    "sensitivities",
    "simulation",
    "tests",
    "utils",
]

# Note: Submodules are not imported automatically to avoid circular imports.
# Import them explicitly when needed:
# from wishart_processes.module import ClassName