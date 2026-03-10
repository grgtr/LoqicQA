from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class LogicQARunRecord:
    """
    Unified schema for LogicQA pipeline run artifacts.
    Serves as input for the hierarchical evaluation framework (Levels 1-4).
    """
    class_name: str
    run_dir: str
    normality_definition: Optional[str] = None
    
    # Level 1: Perception (descriptions of normal images)
    stage1_descriptions: List[Dict] = field(default_factory=list)
    
    # Level 2: Attributes (feature summarization)
    stage2_summary: Optional[Dict] = None
    
    # Level 3: Reasoning (generation, filtering Main-Qs, sub-questions)
    stage3a_questions: Optional[Dict] = None
    stage3b_filtering: List[Dict] = field(default_factory=list)
    stage3c_subqs: Dict[str, Any] = field(default_factory=dict)
    
    # Level 3 and 4: Inference (answers to Sub-Qs and final scores)
    stage4_subq_responses: List[Dict] = field(default_factory=list)
    stage4_image_results: List[Dict] = field(default_factory=list)
