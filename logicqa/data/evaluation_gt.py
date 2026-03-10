from typing import Dict, List, Any

# ==============================================================================
# LEVEL 1: PERCEPTION (Constraint Coverage Rate - CCR)
# ==============================================================================
# LLM-judge will check the presence of each of facts in the generated description.

ATOMIC_CONSTRAINTS: Dict[str, List[str]] = {
    "breakfast_box": [
        "There are exactly two tangerines",
        "There is exactly one nectarine",
        "The tangerines and nectarine are located on the left-hand side",
        "Cereals are present on the right-hand side",
        "A mix of banana chips and almonds is present on the right-hand side",
        "The ratio and relative position of cereals, banana chips, and almonds is fixed"
    ],
    "juice_bottle": [
        "The bottle carries exactly two labels",
        "The first label is attached to the center of the bottle",
        "The fruit icon is positioned exactly at the center of the first label",
        "The second label is attached to the lower part of the bottle",
        "The text '100% Juice' is written on the second label",
        "The bottle is filled with at least 90% of its capacity",
        "The bottle is not filled to 100% (there is a small gap at the top)"
    ],
    "pushpins": [
        "The box is divided into compartments",
        "Each compartment contains exactly one pushpin"
    ],
    "screw_bag": [
        "The bag contains exactly two washers",
        "The bag contains exactly two nuts",
        "The bag contains exactly one long screw",
        "The bag contains exactly one short screw",
        "The bolts (screws) are longer than 3 times the diameter of the washer"
    ],
    "splicing_connectors": [
        "There are exactly two splicing connectors",
        "The two connectors have the same number of cable clamps",
        "The connectors are linked by exactly one cable",
        "The cable is connected to the same position on both connectors (mirror symmetry)",
        "The cable length is longer than the length of the splicing connector terminal block"
    ]
}

# ==============================================================================
# LEVEL 2: ATTRIBUTES (Counting & Spatial Relations)
# ==============================================================================
# Mapping of specific anomaly types (from MVTec LOCO tags) to specific attribute violations.

ATTRIBUTE_GT: Dict[str, Dict[str, Dict[str, Any]]] = {
    "breakfast_box": {
        "missing_tangerine": {"type": "count", "object": "tangerine", "expected": 2, "actual": 1},
        "extra_nectarine": {"type": "count", "object": "nectarine", "expected": 1, "actual": 2},
        "wrong_side_fruit": {"type": "spatial", "object": "fruits", "expected_relation": "left-hand side", "violated": True},
    },
    "splicing_connectors": {
        "missing_cable": {"type": "count", "object": "cable", "expected": 1, "actual": 0},
        "broken_symmetry": {"type": "spatial", "object": "cable", "expected_relation": "connected to the same position (mirror symmetry)", "violated": True},
        "wrong_cable_length": {"type": "spatial", "object": "cable", "expected_relation": "longer than terminal block", "violated": True},
        "different_clamp_count": {"type": "count", "object": "clamps", "expected_relation": "same number on both", "violated": True},
    },
    "screw_bag": {
        "missing_washer": {"type": "count", "object": "washer", "expected": 2, "actual": 1},
        "missing_nut": {"type": "count", "object": "nut", "expected": 2, "actual": 1},
        "extra_short_screw": {"type": "count", "object": "short screw", "expected": 1, "actual": 2},
        "missing_long_screw": {"type": "count", "object": "long screw", "expected": 1, "actual": 0},
    },
    "pushpins": {
        "missing_pushpin": {"type": "count", "object": "pushpin per compartment", "expected": 1, "actual": 0},
        "extra_pushpin": {"type": "count", "object": "pushpin per compartment", "expected": 1, "actual": 2},
    },
    "juice_bottle": {
        "missing_label": {"type": "count", "object": "labels", "expected": 2, "actual": 1},
        "wrong_label_position": {"type": "spatial", "object": "lower label", "expected_relation": "attached to the lower part", "violated": True},
        "wrong_fill_level": {"type": "spatial", "object": "juice level", "expected_relation": "between 90% and 100%", "violated": True},
    }
}
