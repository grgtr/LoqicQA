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

DESCRIBE_PROMPT = """\
This is a {class_name}. Analyze the image and describe the {class_name} in \
detail, including type, color, size (length, width), material, composition, \
quantity, relative location.

Analyze image and extract the core logical rules.
Format your output exactly like this:
1. Components: [List the exact objects that must be present]
2. Quantities: [Exact numbers required, e.g., "exactly two washers"]
3. Spatial Arrangement: [Where items must be located, e.g., "only on the left side"]
4. Visual Appearance: [Specific colors, shapes, or orientations required]

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

SUMMARIZE_PROMPT = """You are an expert industrial quality control analyst.
I will provide you with descriptions of {n_descriptions} NORMAL (defect-free) {class_name} samples.

Your task is to identify the strict INVARIANTS — the characteristics that are completely identical and required across ALL normal samples.

[Descriptions of Normal Samples]
{labeled_descriptions}

[Normality Constraints Provided by Engineer]
{normality_definition}

Analyze the descriptions and extract the core logical rules.
Format your output exactly like this:
1. Components: [List the exact objects that must be present]
2. Quantities: [Exact numbers required, e.g., "exactly two washers"]
3. Spatial Arrangement: [Where items must be located, e.g., "only on the left side"]
4. Visual Appearance: [Specific colors, shapes, or orientations required]

Keep it concise, factual, and strictly based on the provided descriptions."""



# ============================================================
# Stage 3a: Generate candidate main questions
# ============================================================

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

def build_question_slots(n: int) -> str:
    """Generate question slots like (Q1) :\n(Q2) :\n..."""
    return "\n".join(f"(Q{i}) :" for i in range(1, n + 1))

def build_subquestion_slots(n: int) -> str:
    """Generate question slots like (Q1) :\n(Q2) :\n..."""
    return "\n".join(f"Output{i+1}:" for i in range(n))

# ============================================================
# Stage 3b: Generate sub-question variants
# ============================================================

SUBQUESTION_AUGMENT_PROMPT = """You are an AI linguist assisting a Quality Control system.
Your task is to rephrase the given target question into {n_variants} different variations.

Target Question: "{main_question}"

STRICT RULES:
- Must be answered with a simple "Yes" or "No".
- A "Yes" answer MUST mean the image is NORMAL. A "No" answer MUST mean it is an ANOMALY.
- The logical meaning and strictness MUST remain exactly the same.
- If the original question specifies an exact number (e.g., "exactly two"), EVERY variation must include that exact constraint (e.g., "precisely two", "exactly two").
- If the original question specifies a location (e.g., "left side"), EVERY variation must include it.

Output ONLY the rephrased questions, numbered 1 to {n_variants}.

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
