"""All prompt templates for the LogicQA pipeline.
Adopted verbatim from Appendix A of the paper (arXiv:2503.20252).
"""
from __future__ import annotations


# ============================================================
# Stage 1: Describe a normal image
# ============================================================

DESCRIBE_PROMPT = """\
You are a strict industrial quality control inspector.

The following are the Original NORMALITY CONSTRAINTS for this class — treat them as ground truth:
< Normal Constraints for a {class_name} >
{normality_definition}

Now analyze the image of a {class_name} and fill in the form below.
Use ONLY what you can directly observe in the image.
Do NOT speculate, estimate percentages, or add information not visible.
If a section does not apply or cannot be determined from the image, write exactly: N/A

IMPORTANT: You MUST describe EVERY region and compartment of the image.
Do NOT omit any region even if it seems secondary or background.
Every object or area explicitly named in the Normal Constraints MUST be described.

Analyze image and extract the core logical rules.
Format your output exactly like this:

1. Components:
   List every distinct object or part that is visible and must be present.
   Format: short bullet list, one item per line, no sentences.

2. Quantities:
   Exact observed counts for every component.
   Use precise language only: "exactly N", "at least N", "no more than N".
   Do NOT write vague phrases like "some", "a few", "no specific count".
   Format: short bullet list.

3. Spatial Arrangement:
   Where each component is located.
   Use only: left/right/center/top/bottom and relative terms (above, below, adjacent to).
   Format: short bullet list, one constraint per line.

4. Visual Appearance and Fill Level:
   Observable colors, shapes, textures.
   If any container or compartment is visible: is it full, partially full, or empty?
   Are there visible gaps, voids, or empty zones?
   Do NOT estimate percentages unless the image makes it unambiguous.
   Format: short bullet list.

5. Relational and Proportional Constraints:
   Constraints comparing two or more objects (size ratios, length comparisons,
   color-to-count correspondences). Write N/A if none are visible.
   Format: short bullet list or N/A.

6. Symmetry and Connectivity:
   Observable alignment, mirroring, or physical connections between objects.
   Write N/A if none are visible.
   Format: one line or N/A.

7. Per-Slot Completeness:
   If repeating slots or compartments are visible, state what each slot must contain.
   Write N/A if no repeating slots exist.
   Format: one line or N/A.
"""

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

# DESCRIBE_PROMPT = """\
# This is a {class_name}. Analyze the image and describe the {class_name} in \
# detail, including type, color, size (length, width), material, composition, \
# quantity, relative location.

# < Normal Constraints for a {class_name} >
# {normality_definition}
# """


# ============================================================
# Stage 2: Summarize multiple descriptions into normality context
# ============================================================

SUMMARIZE_PROMPT = """\
You are an expert industrial quality control analyst.

Your task is to extract STRICT INVARIANTS — rules that hold true across ALL provided
descriptions without exception.

[Original Normality Constraints — These are ALWAYS correct and ALWAYS take priority]
{normality_definition}

RULE: Every object or region explicitly named in the Original Normality Constraints
MUST appear in your output, regardless of what the descriptions say.
The constraints are ground truth — they override any description.

[Complete component inventory — extracted from ALL Stage 1 descriptions]
Every item below was observed in at least one normal image.
ALL items MUST appear in your section 1. Components output.
Do NOT drop items just because they appear in only some descriptions.
{all_components_hint}

[Descriptions of {n_descriptions} Normal {class_name} Samples]
{labeled_descriptions}

Instructions:
1. The Original Normality Constraints and the complete component inventory are both ground truth.
   Every component listed in the inventory MUST appear in section 1. Components, even if only one description mentions it.
2. ABSENCE of mention ≠ contradiction.
   If some descriptions omit a component and others mention it → KEEP the component
   if it is confirmed by the Original Normality Constraints OR by the component inventory.
3. Write N/A for a point ONLY IF descriptions EXPLICITLY disagree
   (e.g., one says "symmetry required", another says "no symmetry required").
   Silence on a topic is NOT a disagreement.
4. Write ONLY facts. Forbidden phrases: "if applicable", "depending on", "unless stated",
   "approximately", "seems to", "may be", "could be".
5. Each section: maximum 3 bullet points. One fact per bullet. No full sentences.

Output format — fill in each section exactly as shown:

1. Components:
   [bullet list of objects that MUST be present — include ALL items named in Original Constraints]

2. Quantities:
   [bullet list: "exactly N <object>" for every component]

3. Spatial Arrangement:
   [bullet list: one positional rule per line]

4. Visual Appearance and Fill Level:
   [bullet list: colors, shapes, fill states — only confirmed across ALL descriptions]

5. Relational and Proportional Constraints:
   [bullet list of cross-object rules, or N/A]

6. Symmetry and Connectivity:
   [bullet list of alignment/connection rules, or N/A]

7. Per-Slot Completeness:
   [rule per repeating slot/compartment, or N/A]
"""

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

# SUMMARIZE_PROMPT = """\
# {labeled_descriptions}
# Combine the {n_descriptions} descriptions into one by extracting only the \
# "common" features.
# Create a concise summary that reflects the shared characteristics while \
# removing any redundant or unique details.
# """

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

# GENERATE_QUESTIONS_PROMPT = """\
# [ Description of {class_name} ]
# {normality_summary}

# [ Normal Constraints for {class_name} ]
# {normality_definition}

# Using the [ Normal Constraints for {class_name} ] and [ Description of {class_name} ], \
# create several but essential, simple and important questions to determine whether \
# the {class_name} in the image is normal or abnormal. Ensure the questions are \
# only based on visible characteristics, excluding any aspects that cannot be \
# determined from the image. Also, simplify any difficult terms into \
# easy-to-understand questions.
# {question_slots}
# """

GENERATE_QUESTIONS_PROMPT = """You are creating a strict inspection checklist for a Quality Control system.
Based on the summary of a normal {class_name}, generate {n_questions} essential Yes/No questions to detect logical anomalies.

[ Description of {class_name} ]
{normality_summary}

[ Normal Constraints for {class_name} ]
{normality_definition}

Using ONLY the constraints listed in [ Normal Constraints for {class_name} ] and \
[ Description of {class_name} ], create exactly {n_questions} simple and \
important YES/NO questions to determine whether the {class_name} in the image is \
normal or abnormal. Ensure the questions are only based on visible \
characteristics, excluding any aspects that cannot be determined from the \
image. Also, simplify any difficult terms into easy-to-understand questions.
STRICT RULES:
- Must be answered with a simple "Yes" or "No".
- A "Yes" answer MUST mean the image is NORMAL. A "No" answer MUST mean it is an ANOMALY.
- Each question must evaluate exactly ONE logical constraint (e.g., check quantity OR check position, not both).
- Focus strictly on visible, objective characteristics (quantities, colors, specific positions, presence/absence).
- DO NOT use subjective words (e.g., "good", "proper", "normal").
- DO NOT use negative phrasing (e.g., use "Is the box full?" instead of "Is the box not empty?").

Output ONLY the questions, numbered 1 to {n_questions}. Do not add any introductory or concluding text.
{question_slots}
"""

LOCALIZATION_PROMPT = """You are inspecting a {class_name} image.

Locate '{component}' in the image.

Answer in ONE line using this exact format:
  <position> (<count> instance(s))
  or: Not found

Use only spatial terms: left side / right side / center / top / bottom.
Do NOT explain. Output ONLY the location line."""


BULLET_TO_QUESTION_PROMPT = """You are converting a quality control fact into an inspection question.

Fact about a normal {class_name}: {fact}

Convert this fact into a single Yes/No inspection question where:
- "Yes" means the image IS normal (the fact holds)
- "No" means the image is anomalous (the fact is violated)

Rules:
- Output ONE question only, ending with "?"
- Use simple, direct language
- Test exactly ONE observable property
- No negative phrasing (use "Is there X?" not "Is X absent?")
- No subjective words ("good", "proper", "normal")

Output ONLY the question, nothing else."""


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

# SUBQUESTION_AUGMENT_PROMPT = """\
# Generate {n_variants} variations of the following question while keeping the \
# semantic meaning.
# Input: {main_question}
# {subquestion_slots}
# """

SUBQUESTION_AUGMENT_PROMPT = """You are designing visual inspection tests for a quality control system.

A NORMAL image satisfies this constraint:
  "{main_question}"
(Yes = normal / constraint satisfied, No = anomaly detected)

Generate {n_variants} sub-questions that each verify the SAME constraint from a DIFFERENT visual angle.
Do NOT simply rephrase — each question must probe different visual evidence.

Use these probe types (use each at most once, mix them):
  PRESENCE  — "Can you see [object] in [location]?"
  ABSENCE   — "Is [location] empty / missing [object]?"
  FEATURE   — "Do you see [specific visual property: texture, shape, color]?"
  COUNT     — "How many [object] are visible?" (rephrase as Yes/No: "Are there [N] [objects]?")
  SPATIAL   — "Is [object] located on [specific side / position]?"

STRICT RULES:
- Every question must be answerable with only "Yes" or "No".
- "Yes" MUST mean the image is NORMAL. "No" MUST mean an anomaly is present.
- Preserve exact numbers and locations from the original constraint.
- Each question must be genuinely different from all others.

Output ONLY the {n_variants} questions, numbered 1 to {n_variants}.

Format:
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

# TEST_PROMPT = """\
# Question: {question}
# Step 1: List all visible objects in the {class_name} image and their exact counts.
# Step 2: Based on the list above, verify whether the constraint in the question holds.
# Step 3: Conclude with '- Result: Yes' if the constraint holds (image is normal), \
# or '- Result: No' if it is violated (image is anomalous).
# Let's think step by step.
# """

TEST_PROMPT = """You are a strict industrial quality control inspector.
Your task is to inspect a {class_name} and answer a specific constraint question.

{class_context}

Question: {question}

Strict Rules:
1. Base your answer ONLY on direct visual evidence from the image.
2. DO NOT output general knowledge, advice, or hallucinate objects not listed above.
3. Keep your reasoning strictly factual and brief (max 3-4 sentences).

Analyze step-by-step based on the rules, then conclude.
Your response MUST end with exactly:
Result: Yes
or
Result: No"""


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
