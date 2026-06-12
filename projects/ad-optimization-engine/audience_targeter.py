"""
Audience Targeter Module
Implements KNN-based and classification-based lookalike modeling for audience expansion.
Includes reach vs. precision tradeoff controls, exclusion lists, and frequency capping.
"""

from typing import List, Dict, Any, Set
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression


class AudienceTargeter:
    """Manages lookalike audience modeling and targeting filters for ad placement."""

    def __init__(self, seed_profiles: np.ndarray, seed_user_ids: List[str]):
        """
        Initializes the targeter with a seed audience.
        
        Args:
            seed_profiles: Dense feature representation of seed users, shape [N, D]
            seed_user_ids: Unique identifiers corresponding to the seed profiles
        """
        self.seed_profiles = seed_profiles
        self.seed_user_ids = seed_user_ids
        
        # Fit KNN model on the seed profiles for distance-based scoring
        self.knn = NearestNeighbors(n_neighbors=min(5, len(seed_profiles)), metric="cosine")
        self.knn.fit(seed_profiles)
        
        # Classifier for logistic regression lookalike scoring
        self.lookalike_classifier = None
        self.is_trained = False
        
        # Frequency caps and exclusions
        self.exclusions: Set[str] = set()
        self.frequency_ledger: Dict[str, int] = {}

    def train_classifier(self, negative_profiles: np.ndarray):
        """
        Trains a Logistic Regression model to distinguish seeds from general/negative profiles.
        This provides probabilistic lookalike scoring (PU Learning approach).
        """
        # Create labels: 1 for seed users, 0 for negative/general users
        x_train = np.vstack([self.seed_profiles, negative_profiles])
        y_train = np.hstack([np.ones(len(self.seed_profiles)), np.zeros(len(negative_profiles))])
        
        self.lookalike_classifier = LogisticRegression(class_weight="balanced", random_state=42)
        self.lookalike_classifier.fit(x_train, y_train)
        self.is_trained = True

    def add_exclusions(self, user_ids: List[str]):
        """Adds user IDs to the exclusion list (e.g. converted users, opt-outs)."""
        self.exclusions.update(user_ids)

    def record_impression(self, user_id: str):
        """Increments the impression counter for a user (used for frequency capping)."""
        self.frequency_ledger[user_id] = self.frequency_ledger.get(user_id, 0) + 1

    def score_candidates_knn(self, candidate_profiles: np.ndarray) -> np.ndarray:
        """
        Scores candidate users based on average cosine similarity to their top-k nearest seeds.
        Returns similarity scores in range [0, 1].
        """
        # Distances is shape [num_candidates, n_neighbors]
        distances, _ = self.knn.kneighbors(candidate_profiles)
        
        # Cosine distance is in [0, 2]. Cosine similarity is 1.0 - distance.
        # Mean similarity across neighbors:
        mean_distances = np.mean(distances, axis=1)
        similarities = 1.0 - mean_distances
        # Clip to [0, 1] range for safety
        return np.clip(similarities, 0.0, 1.0)

    def score_candidates_classifier(self, candidate_profiles: np.ndarray) -> np.ndarray:
        """
        Scores candidate users using the trained classification model.
        Returns lookalike probability in [0, 1].
        """
        if not self.is_trained:
            # Fall back to KNN scoring if classifier isn't trained
            return self.score_candidates_knn(candidate_profiles)
            
        # Predict probability of class 1 (lookalike seed)
        return self.lookalike_classifier.predict_proba(candidate_profiles)[:, 1]

    def select_audience(
        self, 
        candidates_df: pd.DataFrame, 
        candidate_profiles: np.ndarray, 
        method: str = "knn",
        similarity_threshold: float = 0.7,
        max_frequency: int = 3
    ) -> pd.DataFrame:
        """
        Selects target audience from candidates using similarity filters,
        exclusions, and frequency caps. Allows tuning reach vs. precision.
        
        Args:
            candidates_df: DataFrame containing at least 'user_id'
            candidate_profiles: Profile vectors for candidates, shape [M, D]
            method: Lookalike scoring method ("knn" or "classifier")
            similarity_threshold: Minimum score to target (higher = high precision, lower = high reach)
            max_frequency: Maximum impressions permitted per user before capping out
            
        Returns:
            DataFrame of selected candidates with their lookalike scores.
        """
        assert len(candidates_df) == len(candidate_profiles), "Dimensions of DataFrame and profiles must match."
        
        # Calculate lookalike scores
        if method.lower() == "classifier" and self.is_trained:
            scores = self.score_candidates_classifier(candidate_profiles)
        else:
            scores = self.score_candidates_knn(candidate_profiles)
            
        result_df = candidates_df.copy()
        result_df["lookalike_score"] = scores
        
        # Apply targeting logic (filter by score threshold, exclusions, frequency capping)
        targeted_mask = []
        for _, row in result_df.iterrows():
            uid = str(row["user_id"])
            score = row["lookalike_score"]
            
            # Check exclusions
            if uid in self.exclusions:
                targeted_mask.append(False)
                continue
                
            # Check frequency capping
            if self.frequency_ledger.get(uid, 0) >= max_frequency:
                targeted_mask.append(False)
                continue
                
            # Check similarity threshold (Reach vs. Precision control)
            if score < similarity_threshold:
                targeted_mask.append(False)
                continue
                
            targeted_mask.append(True)
            
        return result_df[targeted_mask].sort_values(by="lookalike_score", ascending=False)


if __name__ == "__main__":
    print("=== AUDIENCE TARGETING LOOKALAKE MODEL TEST ===\n")
    
    # Generate mock seed profiles (dimension=4, e.g. age_scaled, income_scaled, tech_interest, sports_interest)
    # Let's say seed users are tech-savvy high-income users
    rng = np.random.default_rng(42)
    seeds = rng.normal(loc=[0.8, 0.8, 0.9, 0.1], scale=0.1, size=(50, 4))
    seed_ids = [f"seed_{i}" for i in range(50)]
    
    # Generate general/negative profiles (lower tech, lower income)
    general_negatives = rng.normal(loc=[0.3, 0.3, 0.2, 0.5], scale=0.2, size=(100, 4))
    
    # Initialize targeter
    targeter = AudienceTargeter(seeds, seed_ids)
    targeter.train_classifier(general_negatives)
    
    # Generate candidate profiles to score
    candidates = rng.normal(loc=[0.5, 0.5, 0.5, 0.3], scale=0.3, size=(200, 4))
    candidate_ids = [f"user_{i}" for i in range(200)]
    candidates_df = pd.DataFrame({"user_id": candidate_ids, "age": candidates[:, 0], "income": candidates[:, 1]})
    
    # Set exclusions and frequency caps
    targeter.add_exclusions(["user_5", "user_12"]) # converted users
    targeter.record_impression("user_45")
    targeter.record_impression("user_45")
    targeter.record_impression("user_45") # capped out if max_frequency=3
    
    # Evaluate reach vs. precision tradeoff
    for threshold in [0.6, 0.8]:
        selected = targeter.select_audience(
            candidates_df, candidates, method="classifier", similarity_threshold=threshold, max_frequency=3
        )
        print(f"Similarity Threshold: {threshold}")
        print(f"  Audience Size (Reach): {len(selected)} / 200")
        print(f"  Mean Lookalike Score (Precision proxy): {selected['lookalike_score'].mean():.4f}")
        print(f"  Top Target User IDs: {selected['user_id'].head(3).tolist()}")
        print("-" * 40)
