"""EPAL component selection shared by the catalog and acquisition code."""

from typing import Sequence, Union

COMPONENT_ORDER = ('cecp', 'lecp', 'mocp')
COMPONENT_CHOICES = ('cecp', 'lecp', 'mocp', 'cecp+lecp', 'cecp+lecp+mocp')
DEFAULT_COMPONENTS = 'cecp+lecp+mocp'
ERROR_KEYS = {'cecp': 'cls', 'lecp': 'loc', 'mocp': 'miss'}


def normalize_components(value: Union[str, Sequence[str]]) -> str:
    if isinstance(value, str):
        parts = value.lower().split('+')
    elif isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
        parts = [item.lower() for item in value]
    else:
        raise ValueError('EPAL components must be a string or a sequence of names')
    parts = [part.strip() for part in parts]
    if len(set(parts)) != len(parts) or not parts or any(part not in COMPONENT_ORDER for part in parts):
        raise ValueError('Invalid EPAL components: %r. Choose %s' % (value, ', '.join(COMPONENT_CHOICES)))
    normalized = '+'.join(component for component in COMPONENT_ORDER if component in parts)
    if normalized not in COMPONENT_CHOICES:
        raise ValueError('Unsupported EPAL combination: %s. Choose %s' % (normalized, ', '.join(COMPONENT_CHOICES)))
    return normalized
