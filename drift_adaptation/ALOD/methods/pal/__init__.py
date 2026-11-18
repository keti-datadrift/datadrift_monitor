"""PAL active learning method modules."""

from __future__ import annotations

from importlib import import_module


__all__ = [
    'allocate_class_budgets',
    'compute_class_weights',
    'sample_lius_only_from_files',
    'sample_pal_from_files',
    'select_full_pal_images',
    'select_lius_images',
]


def __getattr__(name):
    if name in __all__:
        acquisition = import_module('methods.pal.acquisition')
        return getattr(acquisition, name)
    raise AttributeError("module 'methods.pal' has no attribute %r" % name)


def __dir__():
    return sorted(list(globals()) + __all__)
