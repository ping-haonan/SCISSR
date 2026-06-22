"""SCISSR model package.

Scribble-Conditioned Interactive Surgical Segmentation and Refinement.

The main building blocks live under:
- ``scissr.models``       (Scribble Encoder, SGF, LoRA, ScribbleSam2Memory)
- ``scissr.interactions`` (adaptive scribble generation)

Submodules are imported explicitly by the training / evaluation scripts to
avoid importing heavy optional dependencies at package import time.
"""

__all__ = []
