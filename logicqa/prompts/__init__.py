"""All prompt templates for the LogicQA pipeline.
Adopted verbatim from Appendix A of the paper (arXiv:2503.20252).
"""
from __future__ import annotations


# ============================================================
# Stage 1: Describe a normal image
# ============================================================

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

GENERATE_QUESTIONS_PROMPT = """\
[ Description of {class_name} ]
{normality_summary}

[ Normal Constraints for {class_name} ]
{normality_definition}

Using the [ Normal Constraints for {class_name} ] and \
[ Description of {class_name} ], create several but essential, simple and \
important questions to determine whether the {class_name} in the image is \
normal or abnormal. Ensure the questions are only based on visible \
characteristics, excluding any aspects that cannot be determined from the \
image. Also, simplify any difficult terms into easy-to-understand questions.

(Q1) :
(Q2) :
"""


# ============================================================
# Stage 3b: Generate sub-question variants
# ============================================================

SUBQUESTION_AUGMENT_PROMPT = """\
Generate {n_variants} variations of the following question while keeping the \
semantic meaning.

Input: {main_question}

Output1:
Output2:
Output3:
Output4:
Output5:
"""


# ============================================================
# Stage 4: Test-time — answer a sub-question about a query image
# ============================================================

TEST_PROMPT = """\
Question: {question}
At first, describe {class_name} image.
Your response must end with `- Result: Yes` or `- Result: No`.
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
