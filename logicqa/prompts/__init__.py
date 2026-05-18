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


# ============================================================
# Decomposed Stage 1: per-component description prompts
# ============================================================

IDENTIFY_COMPONENTS_PROMPT = """You are inspecting a {class_name} image.

The following objects MUST be present in a normal {class_name}:
{normality_definition}

Look at the image and list ALL distinct objects you can see.
You MUST include every object named in the normality constraints above,
even if it is hard to see clearly. Add any additional objects you observe.

Output ONLY a bullet list, one item per line, no sentences:
- <object name>
- <object name>
"""

# Count/coverage instruction strings — injected into DESCRIBE_COMPONENT_PROMPT.
# Countable: exact count expected ("exactly N").
# Uncountable: coverage description expected ("a layer", "a handful").
COUNT_INSTR_COUNTABLE = (
    "Count: How many '{component}' do you see? "
    'Use exact language: "exactly N". '
    'If not clearly visible, write "not clearly visible".'
)
COUNT_INSTR_UNCOUNTABLE = (
    "Coverage: '{component}' is an uncountable bulk item — describe its "
    'coverage as "a layer", "a handful", "sparse", etc. '
    "Do NOT count individual pieces."
)
SUMMARIZE_COUNT_INSTR_COUNTABLE = (
    'Count: [exact invariant count across all observations, use exact language: "exactly N".]'
)
SUMMARIZE_COUNT_INSTR_UNCOUNTABLE = (
    "Coverage: '{component}' is an uncountable bulk item — summarize its typical "
    'coverage across all observations, e.g. "a layer", "a handful", "sparse". '
    "Do NOT write an exact count."
)

DESCRIBE_COMPONENT_PROMPT = """You are inspecting a {class_name} image.
Focus ONLY on: '{component}'

[Normality constraints for reference]
{normality_definition}

All components present in this {class_name}:
{all_components_bullet}

Answer exactly these four lines and nothing else:
1. {count_instruction}
2. Position: Where is the '{component}'? Use: left/right/center/top/bottom.
3. Appearance: Color, shape, texture, fill level of '{component}'.
4. Relative size: Compared to the other components listed above, how much space
   does '{component}' occupy?
   Use generic comparisons such as: "largest item", "smaller than <other component>,
   larger than <other component>", "roughly equal to <other component>".

Be brief and factual. Do NOT describe any other component."""

DESCRIBE_RELATIONAL_SLOT_PROMPT = """You are inspecting a {class_name} image.

All components present in this {class_name}:
{all_components_bullet}

Answer the following three sections about cross-component relationships.
Be brief and factual.

5. Relational and Proportional Constraints:
   Which component occupies the most space? The least? Any fixed size ratios between components?

6. Symmetry and Connectivity:
   Are any components strictly separated (e.g., left side vs right side)?
   Are any components required to be parallel, adjacent, or connected?

7. Per-Slot Completeness:
   Are there repeating slots or compartments? If so, what must each slot contain?
   Write N/A if no repeating slots exist."""

SUMMARIZE_COMPONENT_PROMPT = """You are summarizing observations of '{component}' across {n} normal {class_name} images.

[Normality constraints — ground truth]
{normality_definition}

[Observations of '{component}' across {n} images]
{component_observations}

Extract STRICT INVARIANTS — facts true in ALL observations.
If an observation says "not clearly visible", treat it as missing data, not a contradiction.
Forbidden words: "if applicable", "may", "could", "approximately", "seems".

Output exactly four lines:
{count_instruction}
Position: [exact invariant position]
Appearance: [exact invariant visual attributes]
Relative size: [invariant size comparison vs other components, or N/A]"""

SUMMARIZE_RELATIONAL_PROMPT = """You are summarizing cross-component constraints across {n} normal {class_name} images.

[Normality constraints — ground truth]
{normality_definition}

[Cross-component observations across {n} images]
{relational_observations}

Extract STRICT INVARIANTS — facts true in ALL observations.
Forbidden words: "if applicable", "may", "could", "approximately", "seems".

Output in this exact format:

5. Relational and Proportional Constraints:
   Constraints comparing two or more components to each other.
   Include: which component occupies more space, size ratios, count-to-count
   correspondences, any rule that links two components.
   Example: "component_X fills more space than component_Y",
            "count of component_A must equal 2x the count of component_B".
   [bullet list, or N/A]

6. Symmetry and Connectivity:
   Observable alignment, mirroring, or physical connections between components.
   Include: must components be parallel / mirror-symmetric?
   Must they share a boundary or be in separate compartments?
   Example: "component_X on left side, component_Y on right side — strict split".
   [bullet list, or N/A]

7. Per-Slot Completeness:
   If repeating slots or compartments exist, what must each slot contain?
   State the rule per individual slot, not just the total count.
   Example: "each slot must contain exactly one component_X — no empty slots,
            no slots with two or more items".
   [bullet list, or N/A]"""


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

SUBQUESTION_AUGMENT_PROMPT = """You are a quality control assistant verifying images of a packaged product.

The following constraint must hold for a NORMAL image:
  "{main_question}"
(Yes = image is normal, No = anomaly detected)

Write {n_variants} different verification questions about this constraint. Rules:
1. ONLY "Yes" or "No" answers allowed. "Yes" = normal, "No" = anomaly.
2. ONLY use objects and locations that appear in the original constraint. No new objects.
3. ALWAYS phrase questions so that "Yes" means the object/condition IS present or correct.
   WRONG: "Is the left side empty of tangerines?" (Yes = missing = anomaly)
   RIGHT: "Can you see tangerines on the left side?" (Yes = present = normal)
4. Each question must be different from the others and from the original.
5. Preserve exact numbers (e.g., "exactly two", "precisely one") when relevant.

Output ONLY {n_variants} numbered questions, nothing else.

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

[What a NORMAL {class_name} looks like]
{class_context}

[What I observe in the current image under inspection]
{current_image_description}

[Located objects in the current image]
{grounding_context}

Question: {question}

Strict Rules:
1. Compare [What I observe] against [What a NORMAL {class_name} looks like].
2. Base your answer ONLY on the observations listed above.
3. Keep your reasoning strictly factual and brief (max 2-3 sentences).

Conclude with exactly:
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
