"""stator_pipeline — Python interface to the stator mesh construction pipeline."""
from .pipeline import StatorConfig, generate_single, generate_batch

__all__ = ["StatorConfig", "generate_single", "generate_batch"]
