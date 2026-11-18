"""Compatibility helpers for the local MMCV runtime."""


def patch_yapf_verify_arg():
    """Make MMCV 1.x config formatting work with newer yapf releases."""
    try:
        import mmcv.utils.config as mmcv_config
    except ImportError:
        return

    original_format_code = getattr(mmcv_config, 'FormatCode', None)
    if original_format_code is None or getattr(original_format_code, '_alod_compat', False):
        return

    def format_code_compat(*args, **kwargs):
        kwargs.pop('verify', None)
        return original_format_code(*args, **kwargs)

    format_code_compat._alod_compat = True
    mmcv_config.FormatCode = format_code_compat
