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
NORMAL STATE: a single small square glass bottle with a screw cap, standing upright, \
filled with coloured juice and carrying two paper labels.

Labels — presence, identity and order (the primary signals):
- The bottle always carries exactly TWO labels, one above the other.
- UPPER label: shows a single fruit icon (an orange, a banana, an apple, etc.). \
The fruit varies between normal bottles, but a fruit icon is ALWAYS present on the upper label.
- LOWER label: carries the printed text "100% Juice".
- Order is fixed: the fruit-icon label is ALWAYS above the "100% Juice" label. \
The two labels are never swapped, and neither label is ever absent.
- The fruit icon sits roughly in the middle of its label, not pushed into a corner or edge.

Juice fill:
- The bottle is clearly filled with coloured juice — it is NEVER empty or only slightly filled.
- The juice rises to a high level yet leaves a small air gap below the cap: \
it is neither overfilled to the very brim nor noticeably low.

Juice–fruit consistency:
- The juice colour matches the fruit shown on the upper label \
(orange icon → orange juice, banana icon → yellow juice, and so on). \
Juice whose colour disagrees with the depicted fruit is abnormal.

Appearance:
- A small square glass bottle with a screw cap; the juice surface is visible through the glass.

What counts as a logical anomaly: a missing label, a missing fruit icon, swapped label order, \
a fruit icon shoved off-centre, an empty or over-/under-filled bottle, or juice whose colour \
does not match the depicted fruit.""",

    "pushpins": """\
NORMAL STATE: a transparent rectangular plastic box, seen from above, divided into a \
regular grid of small compartments (3 rows × 5 columns, 15 cells in total), with one \
pushpin sitting in each cell.

Count and one-per-cell rule (the central, defining property):
- There are exactly 15 pushpins in total — one in every compartment.
- Each compartment holds EXACTLY ONE pushpin: a cell is never empty and never holds two or more.
- Every pushpin lies in its own cell and does not share a cell with another pushpin.
- The number of pushpins matches the number of compartments exactly: no surplus, no shortfall.

Pushpin appearance:
- All pushpins look alike — the same colour (yellow or orange), the same size, \
each with a round cap and a metal needle.

Layout:
- The 15 cells form a complete, even 3×5 grid; all cells are the same size.

What counts as a logical anomaly: a compartment left empty (a missing pushpin), \
a compartment holding two or more pushpins, or a total pushpin count other than exactly 15.""",

    "screw_bag": """\
NORMAL STATE: a sealed transparent bag holding a small, fixed set of silver metal hardware.

Expected contents — exact set (the primary signal):
- Exactly 2 split-ring washers (flat silver rings, each with a small gap in the ring).
- Exactly 2 hexagonal nuts (six-sided silver pieces).
- Exactly 1 long screw and exactly 1 short screw, each with a hex-socket head and a threaded shaft.
- Six metal items in total and nothing else: every one of the four types is present, \
no type is missing, and there are no extra pieces.

Two screws of different length:
- One screw is CLEARLY longer than the other; the difference is obvious at a glance.
- There is exactly one long screw and exactly one short screw — never two long, never two short.
- Both screws are of ordinary proportions, neither unusually tiny nor unusually long.

Appearance:
- Every item is bare silver/metallic: washers are split rings, nuts are hexagonal, \
screws have a recessed hexagonal socket in the head.

What counts as a logical anomaly: a whole component type missing, the wrong number of any item \
(e.g., one washer, or three nuts), an extra item beyond the six, or the screws being the wrong \
lengths (two long, two short, or a screw of abnormal size).""",

    "splicing_connectors": """\
NORMAL STATE: two push-wire splicing connectors joined by a single cable, lying horizontally \
on a textured background.

Connectors and cable — the core set (primary signals):
- Exactly TWO connectors: transparent blocks with orange lever clamps, one on the LEFT and \
one on the RIGHT.
- The two connectors are the SAME type — the same model, the same number of wire-entry slots \
and the same size. One connector differing in type or size from the other is abnormal.
- Exactly ONE cable joins them through the centre. There is never zero cables, and never \
two or more.

Cable properties:
- The cable is YELLOW. A cable of any other colour (blue, red, etc.) is abnormal.
- The cable is whole and continuous — no cut, nick, break, or exposed wire anywhere along it.

Symmetry:
- The layout is left–right symmetric: the two connectors mirror each other, and the cable \
enters each connector at the same slot position. The cable entering a different slot on one \
side breaks this symmetry and is abnormal.

What counts as a logical anomaly: a missing or extra cable, a cable of the wrong colour, \
a cut or damaged cable, the two connectors being mismatched types/sizes, or an asymmetric \
cable entry.""",
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
