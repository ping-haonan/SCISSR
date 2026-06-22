"""
ScribblePrompt SAM2 Interactions Module

Contains scribble generation strategies and geometry analysis tools.

Key Design Principles:
- Initial annotation: Only positive scribbles (use AdaptiveScribble)
- Iterative correction: Generate correction scribbles based on prediction errors
  - False Negatives → additional positive scribbles
  - False Positives → negative scribbles
"""

from .geometry import GeometryAnalyzer, analyze_geometry

from .scribbles import (
    WarpScribble,
    LineScribble,
    CenterlineScribble,
    ContourScribble,
    WaveSkeletonScribble,
)

from .adaptive_scribble import (
    AdaptiveConfig,
    AdaptiveScribble,
    CorrectionScribbleGenerator,
    adaptive_scribble,
    correction_scribble,
    # Legacy (deprecated)
    NegativeAdaptiveScribble,
    negative_adaptive_scribble,
)

__all__ = [
    # Geometry
    'GeometryAnalyzer',
    'analyze_geometry',
    # Base and Core Scribbles
    'WarpScribble',
    'LineScribble', 
    'CenterlineScribble',
    'ContourScribble',
    'WaveSkeletonScribble',
    # Adaptive Selection (Initial Annotation)
    'AdaptiveConfig',
    'AdaptiveScribble',
    'adaptive_scribble',
    # Correction Scribbles (Iterative Refinement)
    'CorrectionScribbleGenerator',
    'correction_scribble',
    # Legacy (deprecated)
    'NegativeAdaptiveScribble',
    'negative_adaptive_scribble',
]


