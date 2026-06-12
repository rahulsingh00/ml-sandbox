"""
Embedding Generator Module
Generates text embeddings using sentence-transformers and computes cosine similarity.
"""

from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingGenerator:
    """Generates and processes text embeddings using sentence-transformers."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initializes the EmbeddingGenerator with a pre-trained model."""
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generates dense vector embeddings for a given text or list of texts.
        
        Args:
            texts: A single string or a list of strings.
            
        Returns:
            A numpy array of embeddings (shape: [num_texts, embedding_dim]).
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # sentence-transformers encode returns a numpy array
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings

    def normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """
        L2 normalizes a batch of vectors.
        
        Args:
            vectors: Numpy array of shape [n, d].
            
        Returns:
            L2-normalized numpy array.
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1.0, norms)
        return vectors / norms

    def compute_similarity(self, text_a: str, text_b: str) -> float:
        """
        Computes the cosine similarity between two texts.
        
        Args:
            text_a: First input text.
            text_b: Second input text.
            
        Returns:
            Cosine similarity score between 0.0 and 1.0 (or -1.0 and 1.0).
        """
        embeddings = self.embed_texts([text_a, text_b])
        # Normalize vectors
        norm_embeddings = self.normalize_vectors(embeddings)
        
        # Cosine similarity is the dot product of L2-normalized vectors
        similarity = float(np.dot(norm_embeddings[0], norm_embeddings[1]))
        return similarity

if __name__ == "__main__":
    generator = EmbeddingGenerator()
    text1 = "Sightly Enterprises is an industry leader in brand safety and ad placement."
    text2 = "We help brands target the right audience safely and optimize their campaigns."
    text3 = "Unrelated text about baking sourdough bread in a Dutch oven."

    sim_1_2 = generator.compute_similarity(text1, text2)
    sim_1_3 = generator.compute_similarity(text1, text3)

    print("=== EMBEDDING GENERATOR TEST ===")
    print(f"Similarity (1 vs 2): {sim_1_2:.4f}")
    print(f"Similarity (1 vs 3): {sim_1_3:.4f}")
