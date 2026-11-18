from models.fcos.core.modeling.rpn.fcos.fcos import build_fcos


def build_rpn(cfg, in_channels):
    if cfg.MODEL.FCOS_ON:
        return build_fcos(cfg, in_channels)
    raise NotImplementedError("This copied detector package only supports FCOS_ON=True.")
