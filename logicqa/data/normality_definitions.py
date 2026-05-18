"""Normality definitions for each MVTec LOCO AD class.

These definitions are adopted from Bergmann et al. (2022) and used in
Stage 1 (describing normal images) and Stage 3 (generating main questions).
They define the logical constraints that a normal image must satisfy.

Note: For Splicing Connectors and Juice Bottle, the definition varies slightly
depending on cable color / fruit type. A parametrized version is provided.
"""
from __future__ import annotations

from typing import Dict, List, Optional


# --------------------------------------------------------------------------- #
# Fixed normality definitions (from Appendix C.2 of the paper)
# --------------------------------------------------------------------------- #

_NORMALITY_DEFINITIONS: Dict[str, str] = {

    "breakfast_box": """\
COUNTABLE components (exact count matters):
- Tangerines: exactly 2. Always on the LEFT half of the box. Roughly equal size, \
placed side by side or stacked. Medium-sized round citrus fruit.
- Nectarine: exactly 1. Always on the LEFT half of the box, next to the tangerines. \
Smooth round stone fruit, similar size to a tangerine.

UNCOUNTABLE components (describe as a layer or mix — do NOT count individual pieces):
- Cereal mixture: a layer of oat-based granola/cereals. On the RIGHT half, \
typically occupying the upper or larger portion of the right side. \
Occupies MORE space than the banana chips and almonds combined.
- Banana chips: a scattered layer of dried banana slices. In the LOWER portion \
of the RIGHT half, mixed together with almonds beneath the cereal mixture. \
Their relative proportion to almonds varies across images. Do NOT count individual chips.
- Almonds: a scattered layer of whole or halved almonds. In the LOWER portion \
of the RIGHT half, mixed together with banana chips beneath the cereal mixture. \
Their relative proportion to banana chips varies across images. Do NOT count individual almonds.

Spatial layout:
- LEFT half: tangerines + nectarine only.
- RIGHT half: cereal mixture (larger portion) + banana chips and almonds mix (smaller portion).

Relative sizes:
- Cereal mixture occupies more space than banana chips and almonds combined.
- Tangerines are roughly equal in size to each other and similar to the nectarine.
- The left half (fruits) and right half (dry goods) each occupy roughly half the box.""",

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

# --------------------------------------------------------------------------- #
# Improvement 6: Objects that are IMPOSSIBLE for each class.
# Used by LLMJudge to detect hallucinations in Stage 1 descriptions.
# If a description mentions any of these objects, it is likely hallucinated.
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Explicit component lists for decomposed Stage 1/2.
# countable   — exact count matters; VLM is asked "How many?"
# uncountable — bulk/mass items; VLM is asked for coverage, not a count.
# Keys must match class names (lowercase, underscores).
# --------------------------------------------------------------------------- #

NORMALITY_COMPONENTS: Dict[str, Dict[str, List[str]]] = {
    "breakfast_box": {
        "countable":   ["tangerines", "nectarine"],
        "uncountable": ["cereal mixture", "banana chips", "almonds"],
    },
    "screw_bag": {
        "countable":   ["washer", "nut", "long screw", "short screw"],
        "uncountable": [],
    },
    "pushpins": {
        "countable":   ["pushpin"],
        "uncountable": [],
    },
    "juice_bottle": {
        "countable":   ["center label", "lower label"],
        "uncountable": ["bottle"],
    },
    "splicing_connectors": {
        "countable":   ["splicing connector", "cable"],
        "uncountable": [],
    },
}


def get_normality_components(class_name: str):
    """Return (countable, uncountable, all_components) lists for a class."""
    info = NORMALITY_COMPONENTS.get(class_name.lower().replace(" ", "_"), {})
    countable = info.get("countable", [])
    uncountable = info.get("uncountable", [])
    return countable, uncountable, countable + uncountable


IMPOSSIBLE_OBJECTS: Dict[str, list] = {
    "breakfast_box": ["bolt", "screw", "nut", "washer", "cable", "wire", "connector",
                      "pushpin", "pin", "plug", "clamp"],
    "screw_bag":     ["fruit", "cereal", "oat", "chip", "almond", "nectarine",
                      "tangerine", "orange", "mandarin", "banana", "juice", "label",
                      "pushpin", "connector"],
    "pushpins":      ["screw", "bolt", "nut", "washer", "cable", "wire", "fruit",
                      "cereal", "connector", "juice", "label"],
    "splicing_connectors": ["fruit", "cereal", "chip", "almond", "nectarine",
                            "tangerine", "screw", "bolt", "nut", "washer",
                            "pushpin", "juice", "label"],
    "juice_bottle":  ["screw", "bolt", "nut", "washer", "cable", "wire", "connector",
                      "pushpin", "pin", "cereal", "chip", "almond"],
}


SEMANTIC_CONFUSIONS: Dict[str, Dict[str, List[str]]] = {
    "breakfast_box": {
        "nectarine":    ["peach", "plum", "apricot", "berry", "cherry"],
        "tangerine":    ["lemon", "lime", "grapefruit", "citrus"],
        "banana chips": ["raisin", "dried fruit", "date"],
        "almonds":      ["peanut", "cashew", "walnut", "hazelnut"],
    },
    "screw_bag": {
        "washer": ["coin", "ring", "disk", "plate"],
        "nut":    ["bolt head", "cap"],
        "screw":  ["nail", "pin", "spike"],
    },
    "pushpins": {
        "pushpin": ["thumbtack", "tack"],
    },
    "splicing_connectors": {
        "splicing connector": ["terminal block", "junction box"],
        "cable": ["wire rope", "cord"],
    },
    "juice_bottle": {
        "juice":  ["syrup", "water", "soda"],
        "bottle": ["jar", "can", "carton"],
    },
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
