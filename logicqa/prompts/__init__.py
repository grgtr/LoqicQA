"""All prompt templates for the LogicQA pipeline.
Adopted verbatim from Appendix A of the paper (arXiv:2503.20252).
"""
from __future__ import annotations


# ============================================================
# Stage 1: Describe a normal image
# ============================================================

# DESCRIBE_PROMPT = """\
# This is a {class_name}. Analyze the image and describe the {class_name} in \
# detail, including type, color, size (length, width), material, composition, \
# quantity, relative location.

# < Normal Constraints for a {class_name} >
# {normality_definition}
# """

# DESCRIBE_PROMPT = """\
# This is a {class_name}. Analyze the image and describe the {class_name} in \
# detail, including type, color, size (length, width), material, composition, \
# quantity, relative location.

# Analyze image and extract the core logical rules.
# Format your output exactly like this:
# 1. Components: [List every distinct object or part that MUST be present. Note any objects that must be absent or whose presence signals an anomaly. Example: "exactly two splicing connectors, exactly one cable — no extra cables allowed"]
# 2. Quantities: [Exact required counts for every component. Use precise language: "exactly N", "at least N", "no more than N". Example: "exactly two washers, exactly two nuts, one long screw, one short screw"]
# 3. Spatial Arrangement: [Where each component must be located relative to the scene or to other objects. Include absolute positions (left/right/center/top/bottom) and relative positions (above/below/adjacent to another object). Example: "fruits only on the left half; cereals and nuts only on the right half"]
# 4. Visual Appearance and Fill Level: [Specific colors, shapes, surface textures, and orientations required. ALSO describe fill levels, occupancy of containers or regions: [Is any container/bottle/compartment required to be full, partially full, or empty? Are there required gaps, voids, or empty zones that must remain clear? Is the surface required to be uniform, smooth, or free of marks and spots?]. Example: "bottle filled to 90–99% capacity (visible gap at top); surface free of bright spots, streaks, or irregular dark regions"]
# 5. Relational and Proportional Constraints: [Constraints that compare two or more objects to each other NOT absolute values. Include size ratios, length comparisons, color-to-count correspondences. Example: "each screw must be longer than 3x the washer diameter; number of cable clamps must match the cable color code (3 clamps → blue cable)"]
# 6. Symmetry and Connectivity: [How objects must be aligned, mirrored, or connected to each other. Example: "Is mirror or rotational symmetry required?; Must a cable/wire/connector attach to the same position on both ends?; Must certain objects be parallel, perpendicular, or coaxial?; cable must enter the same clamp slot on both connectors (mirror symmetry); connectors must be parallel to each other"]
# 7. Per-Slot Completeness: [If the scene contains repeating slots, cells, or compartments, state the rule that applies to EACH individual slot not just the total count. Example: "each pushpin compartment must contain exactly one pushpin — no empty compartments, no compartments with two or more pushpins"]

# Keep every section concise, factual, and strictly grounded in the provided
# descriptions and constraints. If a section does not apply to this class, write
# "N/A" rather than leaving it blank.

# < Normal Constraints for a {class_name} >
# {normality_definition}
# """

DESCRIBE_PROMPT = """\
This is a {class_name}. Analyze the image and describe the {class_name} in \
detail, including type, color, size (length, width), material, composition, \
quantity, relative location.

< Normal Constraints for a {class_name} >
{normality_definition}
"""


# ============================================================
# Stage 2: Summarize multiple descriptions into normality context
# ============================================================

# SUMMARIZE_PROMPT = """\
# {labeled_descriptions}
# Combine the {n_descriptions} descriptions into one by extracting only the \
# "common" features.
# Create a concise summary that reflects the shared characteristics while \
# removing any redundant or unique details.
# """

# SUMMARIZE_PROMPT = """You are an expert industrial quality control analyst.

# I will provide you with descriptions of {n_descriptions} NORMAL (defect-free) {class_name} samples.

# Your task is to identify the strict INVARIANTS — the characteristics that are completely identical and required across ALL normal samples.

# [Descriptions of Normal Samples]
# {labeled_descriptions}

# [Normality Constraints Provided by Engineer]
# {normality_definition}

# Analyze the descriptions and extract the core logical rules.
# Format your output exactly like this:

# 1. Components: [List every distinct object or part that MUST be present. Note any objects that must be absent or whose presence signals an anomaly. Example: "exactly two splicing connectors, exactly one cable — no extra cables allowed"]
# 2. Quantities: [Exact required counts for every component. Use precise language: "exactly N", "at least N", "no more than N". Example: "exactly two washers, exactly two nuts, one long screw, one short screw"]
# 3. Spatial Arrangement: [Where each component must be located relative to the scene or to other objects. Include absolute positions (left/right/center/top/bottom) and relative positions (above/below/adjacent to another object). Example: "fruits only on the left half; cereals and nuts only on the right half"]
# 4. Visual Appearance and Fill Level: [Specific colors, shapes, surface textures, and orientations required. ALSO describe fill levels, occupancy of containers or regions: [Is any container/bottle/compartment required to be full, partially full, or empty? Are there required gaps, voids, or empty zones that must remain clear? Is the surface required to be uniform, smooth, or free of marks and spots?]. Example: "bottle filled to 90–99% capacity (visible gap at top); surface free of bright spots, streaks, or irregular dark regions"]
# 5. Relational and Proportional Constraints: [Constraints that compare two or more objects to each other NOT absolute values. Include size ratios, length comparisons, color-to-count correspondences. Example: "each screw must be longer than 3x the washer diameter; number of cable clamps must match the cable color code (3 clamps → blue cable)"]
# 6. Symmetry and Connectivity: [How objects must be aligned, mirrored, or connected to each other. Example: "Is mirror or rotational symmetry required?; Must a cable/wire/connector attach to the same position on both ends?; Must certain objects be parallel, perpendicular, or coaxial?; cable must enter the same clamp slot on both connectors (mirror symmetry); connectors must be parallel to each other"]
# 7. Per-Slot Completeness: [If the scene contains repeating slots, cells, or compartments, state the rule that applies to EACH individual slot not just the total count. Example: "each pushpin compartment must contain exactly one pushpin — no empty compartments, no compartments with two or more pushpins"]

# Keep every section concise, factual, and strictly grounded in the provided
# descriptions and constraints. If a section does not apply to this class, write
# "N/A" rather than leaving it blank."""

SUMMARIZE_PROMPT = """\
{labeled_descriptions}
Combine the {n_descriptions} descriptions into one by extracting only the \
"common" features.
Create a concise summary that reflects the shared characteristics while \
removing any redundant or unique details.
"""




# ============================================================
# Stage 3a: Generate candidate main questions
# ============================================================

# GENERATE_QUESTIONS_PROMPT = """
# [ Description of {class_name} ]
# {normality_summary}

# [ Normal Constraints for {class_name} ]
# {normality_definition}

# Using the [ Normal Constraints for {Class} ] and [ Description of {Class} ], create several but essential , simple and important questions to determine whether the {Class} ] in the image is normal or abnormal. Ensure the questions are only based on visible characteristics, excluding any aspects that cannot be determined from the image. Also, simplify any difficult terms into easy-to-understand questions.
# {question_slots}
# """

GENERATE_QUESTIONS_PROMPT = """\
[ Description of {class_name} ]
{normality_summary}

[ Normal Constraints for {class_name} ]
{normality_definition}

Using the [ Normal Constraints for {class_name} ] and [ Description of {class_name} ], \
create several but essential, simple and important questions to determine whether \
the {class_name} in the image is normal or abnormal. Ensure the questions are \
only based on visible characteristics, excluding any aspects that cannot be \
determined from the image. Also, simplify any difficult terms into \
easy-to-understand questions.
{question_slots}
"""

def build_question_slots(n: int) -> str:
    """Generate question slots like (Q1) :\n(Q2) :\n..."""
    return "\n".join(f"(Q{i}) :" for i in range(1, n + 1))

def build_subquestion_slots(n: int) -> str:
    """Generate question slots like (Q1) :\n(Q2) :\n..."""
    return "\n".join(f"Output{i+1}:" for i in range(n))

# ============================================================
# Stage 3b: Generate sub-question variants
# ============================================================
# SUBQUESTION_AUGMENT_PROMPT = """
# Generate five variations of the following question while keeping the semantic meaning.
# Input : {main_question}
# {subquestion_slots}
# """

SUBQUESTION_AUGMENT_PROMPT = """\
Generate {n_variants} variations of the following question while keeping the \
semantic meaning.
Input: {main_question}
{subquestion_slots}
"""


# ============================================================
# Stage 4: Test-time — answer a sub-question about a query image
# ============================================================

# TEST_PROMPT = """\
# Question: {question}
# At first, describe {class_name} image.
# Your response must end with 'Result: Yes' or 'Result: No'.
# Let's think step by step.
# """

TEST_PROMPT = """\
Question: {question}
Step 1: List all visible objects in the {class_name} image and their exact counts.
Step 2: Based on the list above, verify whether the constraint in the question holds.
Step 3: Conclude with '- Result: Yes' if the constraint holds (image is normal), \
or '- Result: No' if it is violated (image is anomalous).
Let's think step by step.
"""



# ============================================================
# Utility: format descriptions for Stage 2
# ============================================================

def format_descriptions(descriptions: list[str], class_name: str) -> str:
    """
    Format descriptions into labeled sections matching the paper's prompt format:
      [ Normal {class_name} Description 1 ]
      ...
    """
    parts = []
    for i, desc in enumerate(descriptions, start=1):
        parts.append(
            f"[ Normal {class_name} Description {i} ]\n{desc.strip()}"
        )
    return "\n\n".join(parts)
