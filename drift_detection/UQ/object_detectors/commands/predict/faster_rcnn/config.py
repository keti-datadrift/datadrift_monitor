from commands.utils.predict_utils import parse_output_config


def parse_faster_rcnn_output_config(output_config):
    return parse_output_config(output_config)


__all__ = ["parse_faster_rcnn_output_config"]
