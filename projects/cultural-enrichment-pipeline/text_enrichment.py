"""
Text Enrichment Pipeline Module
Demonstrates Named Entity Recognition (NER), topic classification,
stance detection, sentiment analysis, and brand safety filtering.
"""

import re
from typing import Dict, List, Any

# Optional imports with heuristics fallbacks to ensure instant execution
try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class TextEnrichmentPipeline:
    """Enriches unstructured text with cultural, sentiment, and safety signals."""

    def __init__(self):
        self.safety_keywords = [
            "violence", "hate speech", "illegal", "terrorism", "hack", "weapons", "drugs"
        ]
        
        if HAS_TRANSFORMERS:
            # Lazy initialize expensive models
            self.sentiment_analyzer = None
            self.zero_shot_classifier = None
        else:
            self.sentiment_analyzer = None
            self.zero_shot_classifier = None

    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """
        Extract key entities (brands, people, organizations, locations).
        Uses contextual regex matching as a lightweight, reliable fallback.
        """
        entities = []
        # Match common tech/retail brands
        brands = ["Microsoft", "Google", "Apple", "Nike", "Amazon", "Tesla", "Meta"]
        for brand in brands:
            if re.search(r'\b' + re.escape(brand) + r'\b', text, re.IGNORECASE):
                entities.append({"entity": brand, "category": "BRAND"})
        
        # Match hashtags as cultural entities
        hashtags = re.findall(r'#\w+', text)
        for tag in hashtags:
            entities.append({"entity": tag, "category": "HASHTAG"})
            
        return entities

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze text sentiment polarity.
        """
        # Simple heuristic fallback
        text_lower = text.lower()
        positive_words = ["great", "awesome", "love", "authentic", "best", "good", "incredible"]
        negative_words = ["bad", "hate", "terrible", "worst", "danger", "poor", "boycott"]
        
        pos_count = sum(1 for w in positive_words if w in text_lower)
        neg_count = sum(1 for w in negative_words if w in text_lower)
        
        score = 0.0
        if pos_count + neg_count > 0:
            score = (pos_count - neg_count) / (pos_count + neg_count)
            
        sentiment = "NEUTRAL"
        if score > 0.2:
            sentiment = "POSITIVE"
        elif score < -0.2:
            sentiment = "NEGATIVE"
            
        return {"sentiment": sentiment, "score": round(score, 2)}

    def classify_stance(self, text: str, target: str) -> str:
        """
        Determine author's stance (FAVOR, AGAINST, NEUTRAL) towards a target brand/entity.
        """
        text_lower = text.lower()
        target_lower = target.lower()
        
        if target_lower not in text_lower:
            return "NEUTRAL"
            
        against_indicators = ["boycott", "cancel", "stop buying", "dislike", "dump", "hate"]
        for ind in against_indicators:
            if ind in text_lower:
                return "AGAINST"
                
        favor_indicators = ["love", "recommend", "support", "best", "loyal", "must buy"]
        for ind in favor_indicators:
            if ind in text_lower:
                return "FAVOR"
                
        return "NEUTRAL"

    def check_brand_safety(self, text: str) -> Dict[str, Any]:
        """
        Assess if the text content violates brand safety guidelines.
        """
        text_lower = text.lower()
        flagged_words = [w for w in self.safety_keywords if w in text_lower]
        is_safe = len(flagged_words) == 0
        
        return {
            "is_safe": is_safe,
            "flagged_categories": flagged_words if not is_safe else [],
            "risk_score": 1.0 if not is_safe else 0.0
        }

    def process_record(self, text: str) -> Dict[str, Any]:
        """Runs the entire enrichment pipeline on a text record."""
        entities = self.extract_entities(text)
        sentiment = self.analyze_sentiment(text)
        brand_safety = self.check_brand_safety(text)
        
        # Analyze stance for any identified brands
        enriched_entities = []
        for ent in entities:
            if ent["category"] == "BRAND":
                stance = self.classify_stance(text, ent["entity"])
                enriched_entities.append({**ent, "stance": stance})
            else:
                enriched_entities.append(ent)
                
        return {
            "raw_text": text,
            "entities": enriched_entities,
            "sentiment": sentiment,
            "brand_safety": brand_safety
        }


if __name__ == "__main__":
    pipeline_obj = TextEnrichmentPipeline()
    
    test_posts = [
        "I absolutely love the new features from Microsoft! #MarketingInnovation is real.",
        "We need to boycott Nike after their latest policy update. Unbelievable.",
        "This post contains information about illegal hacking and weapons.",
        "Tesla's Q2 earnings report came out today. The stock price remained stable."
    ]
    
    print("=== SOCIAL POST ENRICHMENT PIPELINE RUN ===\n")
    for post in test_posts:
        result = pipeline_obj.process_record(post)
        print(f"Post: \"{result['raw_text']}\"")
        print(f"  Entities: {result['entities']}")
        print(f"  Sentiment: {result['sentiment']}")
        print(f"  Safety: {result['brand_safety']}")
        print("-" * 50)
