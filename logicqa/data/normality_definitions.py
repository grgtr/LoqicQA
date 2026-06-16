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
- The left half (fruits) and right half (dry goods) each occupy roughly half the box.

Strict compartment exclusivity:
- LEFT half: tangerines and nectarine ONLY. Cereal mixture, banana chips, and almonds are NEVER present on the left side.
- RIGHT half: cereal mixture, banana chips, and almonds ONLY. Tangerines and nectarine are NEVER present on the right side.""",

    "juice_bottle": """\
COMPONENTS (exact count matters):
- Fruit icon label: exactly 1. Attached to the center or upper portion of the bottle body. \
Contains a clearly visible fruit icon/image centered within the label. \
The fruit type (e.g., orange, banana, apple) varies across normal images, \
but the icon must always be present and centered.
- "100% Juice" label: exactly 1. Attached to the lower portion of the bottle body, \
below the fruit icon label. Contains the text "100% Juice".
- Juice (liquid fill): the bottle body is filled with colored juice. \
Fill level is at least 90% of the bottle's capacity but NOT 100% — \
a small but visible gap must exist between the juice surface and the bottle cap/neck.

Spatial layout:
- Fruit icon label is in the UPPER or CENTER portion of the bottle body.
- "100% Juice" label is BELOW the fruit icon label, in the LOWER portion of the bottle body.
- The fruit icon label is ALWAYS higher on the bottle than the "100% Juice" label. \
They are never swapped.

Visual appearance:
- The bottle is a small square glass bottle with a screw cap.
- The fruit icon is positioned at the CENTER of its label — not shifted left, right, up, or down.
- The juice color matches the fruit depicted on the label \
(e.g., orange juice → orange color, banana juice → yellow color).
- The juice surface is visible inside the bottle; the bottle is never empty or nearly empty.

Fill level constraints:
- Juice fills at least 90% of the bottle body but not 100%.
- A visible gap (air space) exists between the juice surface and the bottle cap.
- The bottle is not filled so high that juice reaches the very neck or cap.

Juice-label consistency:
- The color and type of the juice inside the bottle MUST match the fruit depicted \
on the fruit icon label. A mismatch (e.g., orange juice with a banana icon) is an anomaly.

Strict ordering rule:
- Fruit icon label is ALWAYS above the "100% Juice" label on the bottle. \
Swapped positions are an anomaly.""",

    "pushpins": """\
COMPONENTS (exact count matters):
- Pushpins: exactly 15 in total (one per compartment). Round-capped pins with a metal needle. \
All pushpins are the same color (yellow or orange) and the same size.
- Compartments: exactly 15, arranged in a 3-row by 5-column grid inside a transparent \
rectangular plastic box.
- Separators (dividers): transparent plastic walls forming the internal 3×5 grid. \
All dividers must be present and clearly visible on all four sides of every compartment.

Spatial layout:
- The box forms a complete 3-row × 5-column grid of 15 individual compartments.
- Dividers run both horizontally (separating rows) and vertically (separating columns).
- Every adjacent pair of compartments is separated by a visible plastic wall.

Per-compartment rule (strict):
- Each of the 15 compartments contains EXACTLY ONE pushpin — no more, no less.
- No compartment is empty.
- No compartment contains two or more pushpins.
- Every pushpin is in its own dedicated compartment, not sharing space with another pin.

Divider/separator integrity:
- ALL internal plastic walls between compartments are intact and clearly visible.
- No wall between adjacent compartments is missing, broken, or absent.
- The full 3×5 grid pattern is complete with no merged or open sections.

Visual appearance:
- Pushpins have round colored caps (yellow or orange) and metallic needles pointing downward.
- The box is transparent, allowing all compartments and pushpins to be seen clearly.
- The grid is regular and uniform — all compartments are the same size.""",

    "screw_bag": """\
COUNTABLE components (exact count matters):
- Washers: exactly 2. Small flat split rings with a visible gap/cut (split-ring washers). \
Silver colored. Both washers are the same size.
- Nuts: exactly 2. Hexagonal silver metal pieces. Both nuts are the same size.
- Long screw: exactly 1. Hex socket head screw with a long threaded shaft. \
Clearly longer than the short screw.
- Short screw: exactly 1. Hex socket head screw with a shorter threaded shaft. \
Clearly shorter than the long screw.

Total item count: exactly 6 hardware items in the bag (2 washers + 2 nuts + 1 long screw + 1 short screw).

Size proportions (strict):
- The long screw is CLEARLY longer than the short screw — the length difference is visually obvious.
- The long screw is at least 3× as long as the diameter of a washer.
- The short screw is at least as long as the height of a hexagonal nut (not abnormally tiny).
- Both screws are of standard proportional length — neither screw is abnormally tiny \
nor excessively long relative to the bag size.
- Neither screw extends close to the full height of the bag.

Visual appearance:
- Washers: flat rings with a visible split/gap, silver metal.
- Nuts: six-sided (hexagonal) silver metal pieces.
- Screws: hex socket head (recessed hexagonal socket on top), threaded shaft, silver metal.
- All items are silver/metallic in color.

Completeness rule:
- All four component types MUST be present: washers, nuts, long screw, short screw.
- No component type is missing.
- No extra items beyond the 6 expected pieces.""",

    "splicing_connectors": """\
COUNTABLE components (exact count matters):
- Splicing connectors: exactly 2. Transparent plastic push-wire connectors with orange \
lever clamps. Both connectors are the same type — the same model with the same number \
of wire-entry slots and the same physical size. One connector on the LEFT, one on the RIGHT.
- Cable: exactly 1. A single yellow cable linking the two connectors. \
No extra cables, no missing cable.

Spatial layout:
- Two connectors are placed horizontally on opposite sides, connected by the cable in the center.
- The arrangement is bilaterally symmetric — the left connector and right connector \
are approximate mirror images of each other.
- The cable enters both connectors at the same relative slot position \
(symmetric entry on both sides).

Cable properties:
- Exactly one cable is present — not zero, not two or more.
- The cable is YELLOW in color. A cable of any other color (blue, red, etc.) is an anomaly.
- The cable is undamaged and continuous — no cuts, nicks, breaks, or exposed wire along its length.
- The cable is longer than the width of a single connector terminal block.

Connector matching rule (strict):
- Both connectors must be the same type: same number of orange lever clamps, same height, \
same physical size.
- A mismatch between connector types (e.g., one 2-slot and one 3-slot connector, \
or one 3-slot and one 5-slot) is an anomaly.

Symmetry rule:
- The cable must connect to the same relative slot position on both connectors.
- Inserting the cable into a non-standard slot position on one connector is an anomaly.
- Both connectors must appear as approximate mirror images of each other in the layout.""",
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
