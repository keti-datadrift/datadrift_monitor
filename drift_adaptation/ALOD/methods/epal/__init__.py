"""EPAL method package."""

from __future__ import annotations

__all__ = [
    'sample_epal_from_files',
]


def __getattr__(name):
    if name == 'sample_epal_from_files':
        from methods.epal.acquisition import sample_epal_from_files

        return sample_epal_from_files
    raise AttributeError(name)


def __dir__():
    return sorted(__all__)
