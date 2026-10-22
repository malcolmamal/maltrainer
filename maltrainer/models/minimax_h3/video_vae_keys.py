"""Mapping of the released video VAE checkpoint keys onto the encoder's module names."""

from __future__ import annotations


def map_encoder_key(key: str) -> str | None:
    if not key.startswith(("encoder.", "quant_conv.")):
        return None
    if key.startswith("quant_conv."):
        return key
    rest = key[len("encoder.") :]
    if rest.startswith("down."):
        level, sub = rest[len("down.") :].split(".", 1)
        sub = sub.replace("block.", "resnets.", 1).replace("nin_shortcut.", "conv_shortcut.", 1)
        sub = sub.replace("downsample.", "downsamplers.0.", 1)
        return f"down_blocks.{level}.{sub}"
    if rest.startswith("mid."):
        sub = rest[len("mid.") :].replace("block.", "resnets.", 1).replace("nin_shortcut.", "conv_shortcut.", 1)
        return f"mid_block.{sub}"
    return rest
