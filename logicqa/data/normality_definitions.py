"""Normality definitions for each MVTec LOCO AD class.

These definitions are adopted from Bergmann et al. (2022) and used in
Stage 1 (describing normal images) and Stage 3 (generating main questions).
They define the logical constraints that a normal image must satisfy.

Note: For Splicing Connectors and Juice Bottle, the definition varies slightly
depending on cable color / fruit type. A parametrized version is provided.
"""
from __future__ import annotations

from typing import Dict, Optional


# --------------------------------------------------------------------------- #
# Fixed normality definitions (from Appendix C.2 of the paper)
# --------------------------------------------------------------------------- #

_NORMALITY_DEFINITIONS: Dict[str, str] = {

    "breakfast_box": """\
A normal breakfast box image contains exactly the following items arranged in a cardboard tray:
- One orange juice carton (small)
- One cereal bar (or similar snack bar)
- One banana
- One apple
All items must be present and located within the tray. No item should be missing, duplicated, or replaced by another object. The arrangement may vary slightly, but all four items must be visible within the box.""",

    "juice_bottle": """\
A normal juice bottle image shows a single juice bottle with a label.
The bottle must be upright and fully visible. The label on the bottle comes in different variants
(different fruits: apple, orange, multivitamin, etc.). All bottles share the same shape.
No foreign objects should be present. The cap must be on the bottle.""",

    "pushpins": """\
A normal pushpins image shows a transparent plastic storage box containing black compartments.
Each compartment must contain exactly ONE pushpin.
- No compartment should be empty.
- No compartment should contain more than one pushpin.
- No pushpin should be placed outside of a compartment.
The pushpins are all of the same type (metallic needle, round colored head).""",

    "screw_bag": """\
A normal screw bag image shows a transparent plastic bag containing a specific set of screws and washers.
The normal bag must contain:
- 5 screws of type A (short pan-head screws)
- 5 screws of type B (longer screws)
- 5 washers
All items must be inside the bag. Extra or missing items constitute an anomaly.
No foreign items should be present.""",

    "splicing_connectors": """\
A normal splicing connectors image shows one or more connector blocks.
Each connector block has a fixed number of terminal slots.
Each terminal slot must be occupied by exactly one cable.
The cables come in different colors, and the normal color configuration is consistent per variant.

For a 5-terminal block: cables must be in slots 1 through 5 (left to right).
For a 3-terminal block: cables must be in all 3 slots.
No slot should be empty. No extra cable should be inserted in an already-occupied slot.
Cables must be fully inserted (not partially).""",
}


def get_normality_definition(
    class_name: str,
    variant: Optional[str] = None,
) -> str:
    """
    Return the normality definition string for a given class.

    Args:
        class_name: MVTec LOCO AD class name
                    (e.g., 'breakfast_box', 'juice_bottle').
        variant:    Optional variant info (e.g., cable colour for splicing_connectors).

    Returns:
        Normality definition string.

    Raises:
        KeyError: If the class name is not recognised.
    """
    key = class_name.lower().replace(" ", "_")
    if key not in _NORMALITY_DEFINITIONS:
        raise KeyError(
            f"Unknown class '{class_name}'. "
            f"Available: {list(_NORMALITY_DEFINITIONS.keys())}"
        )
    defn = _NORMALITY_DEFINITIONS[key]
    if variant:
        defn = defn + f"\n\nVariant-specific note: {variant}"
    return defn


def list_classes() -> list[str]:
    """Return all supported MVTec LOCO AD class names."""
    return list(_NORMALITY_DEFINITIONS.keys())


# ------------------------------------------------------------------ #
# Class-level preprocessing flags (from Appendix F)
# ------------------------------------------------------------------ #

#: Classes that require Back Patch Masking (BPM)
BPM_CLASSES = {"screw_bag", "splicing_connectors"}

#: Classes that require Lang-SAM segmentation
LANGSAM_CLASSES = {"pushpins", "splicing_connectors"}
