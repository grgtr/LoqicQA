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
- The breakfast box always contains exactly two tangerines and one nectarine \
that are always located on the left-hand side of the box.
- The ratio and relative position of the cereals and the mix of banana chips \
and almonds on the right-hand side are fixed.""",

    "juice_bottle": """\
- The juice bottle is filled with {fruit} juice and carries exactly two labels.
- The first label is attached to the center of the bottle, with the {fruit} \
icon positioned exactly at the center of the label, clearly indicating the \
type of {fruit} juice.
- The second label is attached to the lower part of the bottle with the text \
"100% Juice" written on it.
- The fill level is the same for each bottle.
- The bottle is filled with at least 90% of its capacity with juice, \
but not 100%.""",

    "pushpins": """\
- Each compartment of the box of pushpins contains exactly one pushpin.""",

    "screw_bag": """\
- A screw bag contains exactly two washers, two nuts, one long screw, \
and one short screw.
- All bolts (screws) are longer than 3 times the diameter of the washer.""",

    "splicing_connectors": """\
- Exactly two splicing connectors with the same number of cable clamps are \
linked by exactly one cable.
- In addition, the number of clamps has a one-to-one correspondence to the \
{color} of the cable.
- The cable must be connected to the same position on both connectors to \
maintain mirror symmetry.
- The cable length is roughly longer than the length of the splicing \
connector terminal block.""",
}

CLASS_INSPECTION_CONTEXTS = {
    "breakfast_box": """
Valid items to look for:
- Fruits: Tangerines (small oranges), Nectarine (smooth round fruit), Apple slices.
- Dry goods: Cereals/Granola (oats), Banana chips (dried slices), Almonds.
Do not confuse nectarines with berries or plums or something else.
Analyze left side and right side separately.
""",
    "juice_bottle": """
Valid items to look for:
- Bottle components: Clear bottle, Cap, Liquid (juice) inside.
- Labels: Exactly two labels (one central, one lower).
- Graphics: Fruit icon on the central label (matching the juice color/type), '100% Juice' text on the lower label.
Check fill levels carefully (must be >90% but not 100%).
""",
    "pushpins": """
Valid items to look for:
- Box: A plastic box divided into compartments.
- Objects: Pushpins of various colors.
Rule: Every single compartment must contain exactly one pushpin. No empty compartments, no multiple pins in one compartment.
""",
    "screw_bag": """
Valid items to look for:
- Container: Transparent plastic bag.
- Hardware: Washers (flat rings), Nuts (hexagonal), Short screws, Long screws.
Rule: Exactly two washers, two nuts, one short screw, and one long screw.
""",
    "splicing_connectors": """
Valid items to look for:
- Hardware: Splicing connectors (transparent with orange levers/clamps).
- Cables: One colored cable linking exactly two connectors.
Rule: Connectors must have the same number of clamps. Cable must maintain mirror symmetry.
"""
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
