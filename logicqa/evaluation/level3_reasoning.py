from typing import List, Dict
from collections import defaultdict

def calculate_subq_consistency(stage4_responses: List[Dict]) -> float:
    """
    Level 3: Confidence estimation (Consistency/Variance) of the model.
    Analyzes the spread of answers within 5 variations of one Sub-Q.
    1.0 = Perfectly consistent (5:0 or 0:5)
    0.0 = Complete confusion (3:2)
    """
    # Group answers by image and main question
    groups = defaultdict(list)
    for resp in stage4_responses:
        img_path = resp.get("image_path", "unknown")
        mq = resp.get("main_question", "unknown")
        answer = resp.get("extracted_answer")
        
        if answer in ["Yes", "No"]:
            groups[(img_path, mq)].append(1 if answer == "Yes" else 0)

    consistencies = []
    
    for (img, mq), answers in groups.items():
        if len(answers) == 0:
            continue
            
        p = sum(answers) / len(answers) # Share of "Yes" answers
        
        # Variance of binomial distribution: p * (1-p)
        # Maximum variance at p=0.4 or 0.6 for 5 questions (3/5) = 0.24
        # To normalize to 0-1, where 1 is super confidence:
        
        # Ideally: p=1.0 or p=0.0 -> var=0 -> consistency=1.0
        # Worst case: p=0.6 or 0.4 -> var=0.24 -> consistency=0.0
        
        variance = p * (1 - p)
        # Normalize so that at variance=0 (5:0) it is 1.0, at variance=0.24 (3:2) it is 0.0
        normalized_consistency = max(0.0, 1.0 - (variance / 0.24))
        
        consistencies.append(normalized_consistency)

    if not consistencies:
        return 0.0
        
    return sum(consistencies) / len(consistencies)
