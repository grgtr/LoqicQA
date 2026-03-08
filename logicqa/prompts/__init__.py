"""All prompt templates for the LogicQA pipeline."""
from __future__ import annotations


# ============================================================
# Stage 1: Describe a normal image
# ============================================================

DESCRIBE_PROMPT = """\
You are an expert visual inspector for industrial quality control.

Below is a normality definition that describes what a correct, defect-free image looks like:
---
{normality_definition}
---

Please analyze the provided image carefully and write a detailed description of it, focusing on:
1. The LOCATION of key objects or components in the image.
2. The QUANTITY of each type of object (e.g., how many screws, bottles, connectors).
3. The APPEARANCE, color, and arrangement of each element.
4. Any structural or spatial relationships between components.

Be specific and factual. Do NOT describe defects or anomalies — describe only what you observe.
"""


# ============================================================
# Stage 2: Summarize multiple descriptions into normality context
# ============================================================

SUMMARIZE_PROMPT = """\
You are summarizing observations from multiple normal (defect-free) industrial images.

Here are {n_descriptions} descriptions of normal images:

{descriptions}

---
Normality definition:
{normality_definition}
---

Please write a concise summary of the SHARED key characteristics that define a normal image in this category. Focus on:
1. What objects/components are consistently present?
2. What are their typical locations and spatial arrangement?
3. What quantity patterns are consistently observed?
4. What appearance features (color, shape) are consistent across normal examples?

Your summary should generalize across all descriptions, capturing the most reliable and consistent normality criteria.
Do not mention image-specific details; only include patterns that appear across multiple images.
"""


# ============================================================
# Stage 3a: Generate candidate main questions
# ============================================================

GENERATE_QUESTIONS_PROMPT = """\
You are creating a quality inspection checklist for detecting LOGICAL ANOMALIES in industrial images.

A logical anomaly is NOT a surface defect (scratch, stain, etc.) — it is a violation of expected rules about:
- The PRESENCE or ABSENCE of required objects
- The QUANTITY of objects (too many or too few)
- The ARRANGEMENT or POSITION of objects

Here is a summary of what a NORMAL image looks like:
---
{normality_summary}
---

And the formal normality definition:
---
{normality_definition}
---

Generate a checklist of {n_questions} binary YES/NO questions that can be used to check whether an image is NORMAL (not anomalous).

Rules for the questions:
- Each question must be answerable with "Yes" (normal) or "No" (anomaly).
- A correct (normal) image should answer "Yes" to every question.
- Questions must target LOGICAL constraints (quantity, presence, arrangement) — not texture or appearance defects.
- Questions must be concise, specific, and independently answerable from the image.
- Number each question on its own line, like: "1. Is there exactly one juice bottle in the image?"

Output ONLY the numbered list of questions, nothing else.
"""


# ============================================================
# Stage 3b: Generate sub-question variants for a main question
# ============================================================

SUBQUESTION_AUGMENT_PROMPT = """\
You are rephrasing a quality inspection question into {n_variants} semantically equivalent variants.

Original question:
"{main_question}"

Generate {n_variants} alternative phrasings of this question that:
- Ask the same thing in different words
- Keep the Yes/No answer format
- A normal image should still answer "Yes" to all variants
- Vary the sentence structure, vocabulary, or perspective

Output ONLY the {n_variants} rephrased questions, numbered 1 to {n_variants}. No explanations.
"""


# ============================================================
# Stage 4: Test-time — answer a sub-question about a query image
# ============================================================

TEST_PROMPT = """\
You are an expert industrial quality inspector.

Examine the provided image carefully and answer the following quality inspection question.

Question: {question}

Instructions:
- Think step by step before giving your answer.
- First describe what you observe in the image relevant to the question.
- Then conclude with "Yes" or "No" on its own line.

Your answer must end with EXACTLY one of:
Yes
No
"""


# ============================================================
# Utility: format descriptions list
# ============================================================

def format_descriptions(descriptions: list[str]) -> str:
    """Format a list of image descriptions into the summarization prompt."""
    return "\n\n".join(
        f"Description {i + 1}:\n{desc.strip()}"
        for i, desc in enumerate(descriptions)
    )
