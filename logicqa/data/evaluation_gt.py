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
        # Counting anomalies (Норма: 1 нектарин, 2 мандарина на левой стороне)
        "0_nectarines_0_tangerines": {"type": "count", "object": "fruits (nectarines and tangerines)", "actual": 0},
        "0_nectarines_1_tangerine": {"type": "count", "object": "tangerines", "actual": 1},
        "0_nectarines_2_tangerines": {"type": "count", "object": "nectarines", "actual": 0},
        "0_nectarines_3_tangerines": {"type": "count", "object": "tangerines", "actual": 3},
        "0_nectarines_4_tangerines": {"type": "count", "object": "tangerines", "actual": 4},
        "1_nectarine_1_tangerine": {"type": "count", "object": "tangerines", "actual": 1},
        "2_nectarines_1_tangerine": {"type": "count", "object": "nectarines", "actual": 2},
        "3_nectarines_0_tangerines": {"type": "count", "object": "nectarines", "actual": 3},
        "missing_almonds": {"type": "count", "object": "almonds", "actual": 0},
        "missing_bananas": {"type": "count", "object": "banana chips", "actual": 0},
        "missing_cereals": {"type": "count", "object": "cereals", "actual": 0},
        "missing_cereals_and_toppings": {"type": "count", "object": "cereals and toppings", "actual": 0},
        "missing_toppings": {"type": "count", "object": "toppings", "actual": 0},
        
        # Spatial / Condition anomalies
        "compartments_swapped": {"type": "spatial", "object": "compartments", "expected_relation": "fruits on the left, cereals on the right", "violated": True},
        "mixed_cereals": {"type": "spatial", "object": "cereals and toppings", "expected_relation": "separated in top and bottom portions on the right side", "violated": True},
        "wrong_ratio": {"type": "spatial", "object": "cereals and toppings", "expected_relation": "fixed ratio and relative position", "violated": True},
        "box_damaged": {"type": "condition", "object": "box", "state": "damaged"},
        "contamination": {"type": "condition", "object": "breakfast box", "state": "contaminated"},
        "fruit_damaged": {"type": "condition", "object": "fruit", "state": "damaged"},
        "overflow": {"type": "spatial", "object": "cereals", "expected_relation": "contained within its compartment without overflowing", "violated": True},
        "underflow": {"type": "count", "object": "cereals", "actual": "too few"},
        "toppings_crushed": {"type": "condition", "object": "toppings", "state": "crushed"}
    },
    
    "juice_bottle": {
        # Counting anomalies (Норма: ровно 2 этикетки)
        "empty_bottle": {"type": "count", "object": "juice", "actual": 0},
        "missing_bottom_label": {"type": "count", "object": "bottom label (100% Juice)", "actual": 0},
        "missing_top_label": {"type": "count", "object": "top label (fruit icon)", "actual": 0},
        "missing_fruit_icon": {"type": "count", "object": "fruit icon on the top label", "actual": 0},
        
        # Spatial anomalies
        "misplaced_label_bottom": {"type": "spatial", "object": "bottom label", "expected_relation": "attached to the lower part of the bottle", "violated": True},
        "misplaced_label_top": {"type": "spatial", "object": "top label", "expected_relation": "attached to the center of the bottle", "violated": True},
        "misplaced_fruit_icon": {"type": "spatial", "object": "fruit icon", "expected_relation": "positioned exactly at the center of the label", "violated": True},
        "rotated_label": {"type": "spatial", "object": "label", "expected_relation": "horizontally aligned", "violated": True},
        "swapped_labels": {"type": "spatial", "object": "labels", "expected_relation": "fruit icon on center, 100% juice on bottom", "violated": True},
        "wrong_fill_level_not_enough": {"type": "spatial", "object": "juice level", "expected_relation": "filled with at least 90% capacity", "violated": True},
        "wrong_fill_level_too_much": {"type": "spatial", "object": "juice level", "expected_relation": "not filled to 100% capacity", "violated": True},
        
        # Condition anomalies
        "contamination": {"type": "condition", "object": "juice bottle", "state": "contaminated"},
        "damaged_label": {"type": "condition", "object": "label", "state": "damaged"},
        "incomplete_fruit_icon": {"type": "condition", "object": "fruit icon", "state": "incomplete"},
        "juice_color": {"type": "condition", "object": "juice", "state": "wrong color"},
        "label_text_incomplete": {"type": "condition", "object": "label text", "state": "incomplete"},
        "unknown_fruit_icon": {"type": "condition", "object": "fruit icon", "state": "unknown"},
        "wrong_juice_type": {"type": "condition", "object": "juice", "state": "does not match label"}
    },
    
    "pushpins": {
        # Counting anomalies (Норма: ровно 1 булавка на ячейку)
        "1_additional_pushpin": {"type": "count", "object": "pushpins in one compartment", "actual": 2},
        "2_additional_pushpins": {"type": "count", "object": "pushpins in one compartment", "actual": 3},
        "missing_pushpin": {"type": "count", "object": "pushpin in one compartment", "actual": 0},
        
        # Spatial / Condition anomalies
        "missing_separator": {"type": "spatial", "object": "compartment separator", "expected_relation": "separates individual pushpins", "violated": True},
        "broken": {"type": "condition", "object": "pushpin", "state": "broken"},
        "color": {"type": "condition", "object": "pushpin", "state": "wrong color"},
        "contamination": {"type": "condition", "object": "box or pushpin", "state": "contaminated"},
        "front_bent": {"type": "condition", "object": "pushpin needle", "state": "bent"}
    },
    
    "splicing_connectors": {
        # Counting anomalies (Норма: 2 коннектора, 1 кабель)
        "missing_cable": {"type": "count", "object": "cable", "actual": 0},
        "extra_cable": {"type": "count", "object": "cable", "actual": 2},
        "missing_connector": {"type": "count", "object": "splicing connector", "actual": 1},
        
        # Spatial anomalies
        "flipped_connector": {"type": "spatial", "object": "connector", "expected_relation": "positioned parallel to maintain mirror symmetry", "violated": True},
        "wrong_cable_location": {"type": "spatial", "object": "cable", "expected_relation": "connected to the same position on both connectors (mirror symmetry)", "violated": True},
        "cable_not_plugged": {"type": "spatial", "object": "cable", "expected_relation": "fully plugged into the connector", "violated": True},
        "open_lever": {"type": "spatial", "object": "lever", "expected_relation": "closed tightly", "violated": True},
        "wrong_connector_type_3_2": {"type": "spatial", "object": "connectors", "expected_relation": "both connectors must have the same number of clamps", "violated": True},
        "wrong_connector_type_5_2": {"type": "spatial", "object": "connectors", "expected_relation": "both connectors must have the same number of clamps", "violated": True},
        "wrong_connector_type_5_3": {"type": "spatial", "object": "connectors", "expected_relation": "both connectors must have the same number of clamps", "violated": True},
        
        # Condition anomalies
        "broken_cable": {"type": "condition", "object": "cable", "state": "broken"},
        "broken_connector": {"type": "condition", "object": "connector", "state": "broken"},
        "cable_color": {"type": "condition", "object": "cable", "state": "wrong color for the clamp count"},
        "cable_cut": {"type": "condition", "object": "cable", "state": "cut"},
        "color": {"type": "condition", "object": "connector", "state": "wrong color"},
        "contamination": {"type": "condition", "object": "connectors", "state": "contaminated"},
        "unknown_cable_color": {"type": "condition", "object": "cable", "state": "unknown color"}
    },
    
    "screw_bag": {
        # Counting anomalies (Норма: 2 шайбы, 2 гайки, 1 длинный винт, 1 короткий винт)
        "1_additional_long_screw": {"type": "count", "object": "long screw", "actual": 2},
        "1_additional_nut": {"type": "count", "object": "nut", "actual": 3},
        "1_additional_short_screw": {"type": "count", "object": "short screw", "actual": 2},
        "1_additional_washer": {"type": "count", "object": "washer", "actual": 3},
        "1_missing_long_screw": {"type": "count", "object": "long screw", "actual": 0},
        "1_missing_nut": {"type": "count", "object": "nut", "actual": 1},
        "1_missing_short_screw": {"type": "count", "object": "short screw", "actual": 0},
        "1_missing_washer": {"type": "count", "object": "washer", "actual": 1},
        "2_additional_nuts": {"type": "count", "object": "nut", "actual": 4},
        "2_additional_washers": {"type": "count", "object": "washer", "actual": 4},
        "2_missing_nuts": {"type": "count", "object": "nut", "actual": 0},
        "2_missing_washers": {"type": "count", "object": "washer", "actual": 0},
        
        # Spatial / Condition anomalies
        "screw_too_long": {"type": "spatial", "object": "screw length", "expected_relation": "matches standard length", "violated": True},
        "screw_too_short": {"type": "spatial", "object": "screw length", "expected_relation": "matches standard length (longer than 3 times washer diameter)", "violated": True},
        "1_very_short_screw": {"type": "spatial", "object": "screw length", "expected_relation": "matches standard length", "violated": True},
        "2_very_short_screws": {"type": "spatial", "object": "screw length", "expected_relation": "matches standard length", "violated": True},
        "bag_broken": {"type": "condition", "object": "bag", "state": "broken"},
        "color": {"type": "condition", "object": "hardware item", "state": "wrong color"},
        "contamination": {"type": "condition", "object": "bag", "state": "contaminated"},
        "part_broken": {"type": "condition", "object": "hardware item", "state": "broken"}
    }
}
