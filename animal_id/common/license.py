"""Whether backbone weights or training images can ship in a model bound for Immich."""

from enum import Enum


class LicenseTier(Enum):
    """Shippability of weights or data.

    For backbones, timm reports the weight license per tag
    (``timm.get_pretrained_cfg(tag).license``); the architecture being Apache says
    nothing about the weights — every ConvNeXt-V2 tag is CC-BY-NC.
    """

    PERMISSIVE = "permissive"  # Apache-2.0 / BSD / CC-BY — shippable.
    NONCOMMERCIAL = "noncommercial"  # CC-BY-NC — reference ceiling only.
    ENCUMBERED = "encumbered"  # Bespoke or unstated terms, needs a license decision.


def license_tier(license: str) -> LicenseTier:
    """Tier of a dataset license string as the source states it (``"CC BY 4.0"``)."""
    words = [w for w in license.upper().replace("-", " ").split() if not w[0].isdigit()]
    if "NC" in words:
        return LicenseTier.NONCOMMERCIAL
    # ShareAlike and NoDerivs stay encumbered: whether they bind trained weights
    # is unsettled, so each needs a decision rather than a default.
    if words in (["CC", "BY"], ["CC0"]):
        return LicenseTier.PERMISSIVE
    return LicenseTier.ENCUMBERED
