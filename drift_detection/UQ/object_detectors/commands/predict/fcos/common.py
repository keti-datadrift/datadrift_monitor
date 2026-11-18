def unpack_fcos_model_output(model_output):
    raw_prediction = model_output[0] if isinstance(model_output, (tuple, list)) else model_output
    raw_logits = model_output[1] if isinstance(model_output, (tuple, list)) and len(model_output) > 1 else None
    raw_indices = model_output[2] if isinstance(model_output, (tuple, list)) and len(model_output) > 2 else None
    return raw_prediction, raw_logits, raw_indices


def select_fcos_post_nms(detector, raw_prediction, raw_logits=None, raw_indices=None):
    return detector.select_post_nms_predictions(
        raw_prediction,
        logits=raw_logits,
        raw_indices=raw_indices,
    )
