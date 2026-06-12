"""
Clustering Module
Demonstrates representation learning and unsupervised clustering (K-Means)
to discover emerging cultural trends from unstructured social records.
"""

from typing import List, Dict
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


class CulturalTrendClusterer:
    """Vectorizes text data and groups it into trending thematic clusters."""

    def __init__(self, num_clusters: int = 3):
        self.num_clusters = num_clusters
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
        self.model = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')

    def fit_and_predict(self, documents: List[str]) -> List[int]:
        """Vectorizes documents and fits K-Means model, returning cluster assignments."""
        tfidf_matrix = self.vectorizer.fit_transform(documents)
        return self.model.fit_predict(tfidf_matrix).tolist()

    def get_cluster_keywords(self) -> Dict[int, List[str]]:
        """Identifies top keywords associated with each cluster center."""
        terms = self.vectorizer.get_feature_names_out()
        centroids = self.model.cluster_centers_
        
        cluster_keywords = {}
        for idx in range(self.num_clusters):
            # Sort term indices by descending centroid score value
            top_term_indices = centroids[idx].argsort()[::-1][:5]
            keywords = [terms[i] for i in top_term_indices]
            cluster_keywords[idx] = keywords
            
        return cluster_keywords


if __name__ == "__main__":
    social_posts = [
        "An adtech startup announces new optimization tools for real-time digital advertising.",
        "Meta launches next-gen social intelligence model to track ad performance.",
        "Severe storm warnings issued for the East Coast this weekend.",
        "Hurricane updates: local alerts advise residents to stay indoors.",
        "The new electric vehicle models feature long battery life and fast charging.",
        "Tesla's autonomous autopilot systems show massive safety improvements."
    ]
    
    clusterer = CulturalTrendClusterer(num_clusters=3)
    assignments = clusterer.fit_and_predict(social_posts)
    keywords = clusterer.get_cluster_keywords()
    
    print("=== CULTURAL CLUSTERING & TREND DISCOVERY RUN ===\n")
    print(f"Total documents analyzed: {len(social_posts)}\n")
    
    # Print grouped results
    grouped_posts: Dict[int, List[str]] = {i: [] for i in range(3)}
    for post, cluster_id in zip(social_posts, assignments):
        grouped_posts[cluster_id].append(post)
        
    for cluster_id in range(3):
        print(f"🚨 CLUSTER {cluster_id}")
        print(f"  Key Words: {keywords[cluster_id]}")
        print("  Posts in Cluster:")
        for post in grouped_posts[cluster_id]:
            print(f"    - \"{post}\"")
        print()
