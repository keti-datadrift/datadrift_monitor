"""Pure method-runtime helpers shared across active-learning methods.

Import helpers from their concrete modules, for example
``methods.common.coco_pool`` or ``methods.common.selection``. This package does
not re-export helper functions so call sites keep the dependency boundary
explicit.
"""
