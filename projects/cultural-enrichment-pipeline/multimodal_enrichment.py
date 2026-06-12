"""
Multimodal Enrichment Module
Simulates CLIP-based contrastive language-image alignment
for brand safety screening and contextual ad-targeting matching.
"""

from typing import Dict, List, Any
import numpy as np

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class MultimodalEnrichmentPipeline:
    """Enriches image signals and screens visual content for brand safety."""

    def __init__(self):
        # Category vectors mock for zero-shot prediction demo
        # Maps textual labels to hypothetical visual centroids in CLIP space
        self.safety_labels = ["safe environment", "hate speech banner", "physical violence", "weapon or firearm"]
        
    def extract_image_embedding(self, image_path: str) -> np.ndarray:
        """
        Simulates extracting a dense 512-dimensional embedding from an image.
        In production, this runs a Vision Transformer (ViT) image encoder.
        """
        # Seed generator based on filename length to get deterministic mock embeddings
        seed = len(image_path)
        rng = np.random.default_rng(seed)
        embedding = rng.normal(size=512)
        # Normalize the embedding vector
        return embedding / np.linalg.norm(embedding)

    def extract_text_embeddings(self, labels: List[str]) -> Dict[str, np.ndarray]:
        """
        Simulates extracting dense embeddings for text candidate labels.
        In production, this runs the CLIP Text Transformer.
        """
        text_embeddings = {}
        for idx, label in enumerate(labels):
            # Deterministic generation for mocks
            rng = np.random.default_rng(idx + 100)
            emb = rng.normal(size=512)
            text_embeddings[label] = emb / np.linalg.norm(emb)
        return text_embeddings

    def evaluate_image_safety(self, image_path: str) -> Dict[str, Any]:
        """
        Computes zero-shot similarity between image embedding and safety labels.
        If similarity to unsafe labels is above threshold, marks as unsafe.
        """
        img_emb = self.extract_image_embedding(image_path)
        txt_embs = self.extract_text_embeddings(self.safety_labels)
        
        scores = {}
        for label, txt_emb in txt_embs.items():
            # Cosine similarity (dot product of normalized vectors)
            similarity = float(np.dot(img_emb, txt_emb))
            scores[label] = round(similarity, 4)
            
        # Classify based on highest similarity
        max_label = max(scores, key=scores.get)
        is_safe = max_label == "safe environment"
        
        return {
            "image_path": image_path,
            "is_safe": is_safe,
            "primary_visual_concept": max_label,
            "similarity_scores": scores
        }


if __name__ == "__main__":
    pipeline_obj = MultimodalEnrichmentPipeline()
    
    mock_images = [
        "assets/images/happy_family_eating_cereal.jpg",
        "assets/images/protest_violence_street.png",
        "assets/images/rifle_on_wooden_table.jpg"
    ]
    
    print("=== MULTIMODAL IMAGE ENRICHMENT RUN ===\n")
    for img in mock_images:
        safety_report = pipeline_obj.evaluate_image_safety(img)
        print(f"Image Path: {safety_report['image_path']}")
        print(f"  Brand Safe: {safety_report['is_safe']}")
        print(f"  Primary Category: {safety_report['primary_visual_concept']}")
        print(f"  Score Matrix: {safety_report['similarity_scores']}")
        print("-" * 50)
