"""Shared evaluation-only interventions for D-HRQS query admission."""


def configure_standard_queries(model):
    """Restore ordinary queries and remove the extra S8 level from encoders."""
    hrqs_decoders = [
        module
        for module in model.modules()
        if getattr(module, "hrqs_enabled", False)
    ]
    if len(hrqs_decoders) != 1:
        raise ValueError(
            "standard_queries requires exactly one active HRQS decoder, "
            f"found {len(hrqs_decoders)}"
        )
    hrqs_decoders[0].hrqs_enabled = False

    def discard_extra_s8_level(_module, inputs):
        features = inputs[0]
        return (features[1:],) + tuple(inputs[1:])

    encoders = [model.encoder]
    if getattr(model, "thermal_encoder", None) is not None:
        encoders.append(model.thermal_encoder)
    return [
        encoder.register_forward_pre_hook(discard_extra_s8_level)
        for encoder in encoders
    ]

