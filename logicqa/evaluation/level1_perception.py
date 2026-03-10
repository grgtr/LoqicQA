import torch
from PIL import Image
from typing import List, Dict
from transformers import CLIPProcessor, CLIPModel
from logicqa.evaluation.llm_judge import LLMJudge

CLIP_MAX_TOKENS = 77

class PerceptionEvaluator:
    def __init__(self, device: str = "cuda"):
        print("[PerceptionEvaluator] Loading CLIP model...")
        self.device = device
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def _chunk_text(self, text: str, max_tokens: int = CLIP_MAX_TOKENS) -> List[str]:
        """
        Splits long text into chunks that do not exceed max_tokens.
        Splitting occurs by words to avoid breaking words.
        """
        words = text.split()
        chunks = []
        current_chunk_words = []
            
        for word in words:
            candidate = " ".join(current_chunk_words + [word])
            token_count = len(self.clip_processor.tokenizer.encode(candidate))
            
            if token_count > max_tokens:
                if current_chunk_words:
                    chunks.append(" ".join(current_chunk_words))
                current_chunk_words = [word]
            else:
                current_chunk_words.append(word)
        
        if current_chunk_words:
            chunks.append(" ".join(current_chunk_words))
        
        return chunks if chunks else [text[:200]]


    def calculate_clip_score(self, image_path: str, description: str) -> float:
        """
        Calculates the cosine similarity between an image and a text.
        
        For long descriptions, splits the text into chunks of <= 77 tokens,
        calculates the score for each chunk and returns the maximum.
        Maximum instead of average, as we need to find the most
        relevant fragment of the description to the image.
        """
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"[PerceptionEvaluator] Cannot open image {image_path}: {e}")
            return 0.0
        
        # Check token length of the tokenized text
        token_count = len(self.clip_processor.tokenizer.encode(description))
        
        if token_count <= CLIP_MAX_TOKENS:
            # Text fits — process directly with truncation as a fallback
            inputs = self.clip_processor(
                text=[description],
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=CLIP_MAX_TOKENS
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
            return outputs.logits_per_image.item()
        
        # Text is too long — split into chunks
        chunks = self._chunk_text(description, max_tokens=CLIP_MAX_TOKENS - 2)  # -2 for [CLS] and [SEP]
        
        if not chunks:
            return 0.0
        
        chunk_scores = []
        
        for chunk in chunks:
            inputs = self.clip_processor(
                text=[chunk],
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=CLIP_MAX_TOKENS
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
            chunk_scores.append(outputs.logits_per_image.item())
        
        # Return maximum over chunks: find the most relevant fragment of the description
        raw_score = max(chunk_scores)

        normalized_score = max(0.0, (raw_score / 100.0) * 2.5) * 100.0
        
        return normalized_score

    def calculate_ccr(self, description: str, gt_constraints: List[str], judge: LLMJudge) -> Dict:
        """
        Calculate Constraint Coverage Rate.
        Returns a dictionary with the results of the check and the final percentage.
        """
        if not gt_constraints:
            return {"ccr_score": 0.0, "details": {}}
            
        results = judge.evaluate_ccr(description, gt_constraints)
        
        covered = sum(1 for v in results.values() if v is True)
        score = covered / len(gt_constraints) if gt_constraints else 0.0
        
        return {
            "ccr_score": score,
            "details": results
        }
