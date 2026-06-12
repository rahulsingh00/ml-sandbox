"""
Text Enrichment Pipeline Module
Demonstrates Named Entity Recognition (NER), topic classification,
stance detection, sentiment analysis, and brand safety filtering.
"""

import re
from typing import Dict, List, Any, Optional

# Optional imports with heuristic fallbacks to ensure instant execution
try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class TextEnrichmentPipeline:
    """Enriches unstructured text with cultural, sentiment, and safety signals using HuggingFace models."""

    def __init__(self):
        self.safety_keywords = [
            "violence", "hate speech", "illegal", "terrorism", "hack", "weapons", "drugs"
        ]
        self.known_brands = ["Microsoft", "Google", "Apple", "Nike", "Amazon", "Tesla", "Meta"]
        self.topics = ["Tech", "Sports", "Politics", "Entertainment", "Finance", "Healthcare", "Fashion", "Marketing"]
        
        # Lazy initialize expensive pipeline models on first use
        self.ner_pipeline = None
        self.sentiment_pipeline = None
        self.zero_shot_pipeline = None
        self.safety_pipeline = None

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract key entities (brands, people, organizations, locations).
        Runs HuggingFace NER model first and falls back to regex matching if unavailable.
        """
        entities = []
        
        if HAS_TRANSFORMERS:
            try:
                if self.ner_pipeline is None:
                    # Use simple aggregation strategy to automatically merge B-/I- subword tokens
                    self.ner_pipeline = pipeline(
                        "token-classification", 
                        model="dslim/bert-base-NER", 
                        aggregation_strategy="simple"
                    )
                hf_ents = self.ner_pipeline(text)
                for ent in hf_ents:
                    category = ent["entity_group"]
                    word = ent["word"]
                    
                    # Map standard NER categories to more descriptive names or BRAND
                    if word.lower() in [b.lower() for b in self.known_brands]:
                        category = "BRAND"
                    elif category == "ORG":
                        category = "ORGANIZATION"
                    elif category == "PER":
                        category = "PERSON"
                    elif category == "LOC":
                        category = "LOCATION"
                    
                    entities.append({
                        "entity": word,
                        "category": category,
                        "score": round(float(ent["score"]), 3)
                    })
            except Exception:
                # If model loading/inference fails, proceed to fallback
                pass

        # Exclude duplicate entries when combining pipeline + regex fallbacks
        existing_entities = {e["entity"].lower() for e in entities}

        # Regex-based brand detection fallback
        for brand in self.known_brands:
            if brand.lower() not in existing_entities:
                if re.search(r'\b' + re.escape(brand) + r'\b', text, re.IGNORECASE):
                    entities.append({"entity": brand, "category": "BRAND", "score": 1.0})
                    existing_entities.add(brand.lower())

        # Match hashtags as cultural entities
        hashtags = re.findall(r'#\w+', text)
        for tag in hashtags:
            if tag.lower() not in existing_entities:
                entities.append({"entity": tag, "category": "HASHTAG", "score": 1.0})
                existing_entities.add(tag.lower())
            
        return entities

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze text sentiment polarity using CardiffNLP's Twitter RoBERTa sentiment model.
        Falls back to keyword counting.
        """
        if HAS_TRANSFORMERS:
            try:
                if self.sentiment_pipeline is None:
                    self.sentiment_pipeline = pipeline(
                        "text-classification", 
                        model="cardiffnlp/twitter-roberta-base-sentiment-latest"
                    )
                res = self.sentiment_pipeline(text)[0]
                label = res["label"].upper()
                score = float(res["score"])
                
                # Twitter RoBERTa labels: POSITIVE, NEGATIVE, NEUTRAL
                if "POS" in label:
                    sentiment = "POSITIVE"
                    norm_score = score
                elif "NEG" in label:
                    sentiment = "NEGATIVE"
                    norm_score = -score
                else:
                    sentiment = "NEUTRAL"
                    norm_score = 0.0
                    
                return {"sentiment": sentiment, "score": round(norm_score, 2)}
            except Exception:
                pass

        # Simple heuristic fallback
        text_lower = text.lower()
        positive_words = ["great", "awesome", "love", "authentic", "best", "good", "incredible", "innovation"]
        negative_words = ["bad", "hate", "terrible", "worst", "danger", "poor", "boycott", "unbelievable"]
        
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
        Uses BART Large MNLI Zero-Shot Classification.
        """
        if HAS_TRANSFORMERS:
            try:
                if self.zero_shot_pipeline is None:
                    self.zero_shot_pipeline = pipeline(
                        "zero-shot-classification", 
                        model="facebook/bart-large-mnli"
                    )
                candidate_labels = ["in favor of", "against", "neutral toward"]
                hypothesis = f"This text is {{}} toward {target}."
                res = self.zero_shot_pipeline(text, candidate_labels, hypothesis_template=hypothesis)
                best_label = res["labels"][0]
                
                if best_label == "in favor of":
                    return "FAVOR"
                elif best_label == "against":
                    return "AGAINST"
                else:
                    return "NEUTRAL"
            except Exception:
                pass

        # Heuristic fallback
        text_lower = text.lower()
        target_lower = target.lower()
        
        if target_lower not in text_lower:
            return "NEUTRAL"
            
        against_indicators = ["boycott", "cancel", "stop buying", "dislike", "dump", "hate", "unbelievable"]
        for ind in against_indicators:
            if ind in text_lower:
                return "AGAINST"
                
        favor_indicators = ["love", "recommend", "support", "best", "loyal", "must buy", "great", "awesome"]
        for ind in favor_indicators:
            if ind in text_lower:
                return "FAVOR"
                
        return "NEUTRAL"

    def classify_topic(self, text: str, topics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Classify the topic of the text against a taxonomy using Zero-Shot BART MNLI.
        """
        if topics is None:
            topics = self.topics

        if HAS_TRANSFORMERS:
            try:
                if self.zero_shot_pipeline is None:
                    self.zero_shot_pipeline = pipeline(
                        "zero-shot-classification", 
                        model="facebook/bart-large-mnli"
                    )
                res = self.zero_shot_pipeline(text, topics)
                topic_scores = {topic: round(float(score), 3) for topic, score in zip(res["labels"], res["scores"])}
                best_topic = res["labels"][0]
                return {
                    "primary_topic": best_topic,
                    "topic_distribution": topic_scores
                }
            except Exception:
                pass

        # Heuristic fallback for topic taxonomy
        text_lower = text.lower()
        topic_keywords = {
            "Tech": ["tech", "software", "hardware", "ai", "computer", "microsoft", "google", "apple", "tesla", "meta", "amazon", "server", "code", "hacking"],
            "Sports": ["sports", "game", "match", "run", "player", "ball", "nike", "stadium", "athlete"],
            "Politics": ["politics", "policy", "government", "vote", "election", "law"],
            "Entertainment": ["entertainment", "movie", "song", "music", "actor", "show"],
            "Finance": ["finance", "stock", "earnings", "price", "revenue", "money", "market", "q2"],
            "Healthcare": ["health", "hospital", "medicine", "doctor", "disease"]
        }
        
        scores = {}
        for topic, keywords in topic_keywords.items():
            scores[topic] = sum(1 for kw in keywords if kw in text_lower)
            
        primary = max(scores, key=lambda k: scores[k]) if any(scores.values()) else "Neutral"
        total = sum(scores.values())
        if total > 0:
            dist = {k: round(v/total, 2) for k, v in scores.items()}
        else:
            dist = {k: 0.0 for k in topic_keywords}
            primary = "Neutral"
            
        return {
            "primary_topic": primary,
            "topic_distribution": dist
        }

    def check_brand_safety(self, text: str) -> Dict[str, Any]:
        """
        Assess if the text content violates brand safety guidelines.
        Complements keyword blacklist with Unitary Toxic BERT model inference.
        """
        text_lower = text.lower()
        flagged_words = [w for w in self.safety_keywords if w in text_lower]
        keyword_risk = 1.0 if flagged_words else 0.0
        
        toxic_score = 0.0
        if HAS_TRANSFORMERS:
            try:
                if self.safety_pipeline is None:
                    # unitary/toxic-bert returns scores for multi-label toxicity
                    self.safety_pipeline = pipeline(
                        "text-classification", 
                        model="unitary/toxic-bert", 
                        return_all_scores=True
                    )
                res = self.safety_pipeline(text)[0]
                toxic_score = max(float(label_dict["score"]) for label_dict in res)
            except Exception:
                pass
                
        risk_score = max(keyword_risk, toxic_score)
        is_safe = risk_score < 0.5
        
        flagged_categories = flagged_words.copy()
        if toxic_score >= 0.5:
            flagged_categories.append("model_flagged_toxicity")
            
        return {
            "is_safe": is_safe,
            "flagged_categories": flagged_categories,
            "risk_score": round(risk_score, 2)
        }

    def process_record(self, text: str) -> Dict[str, Any]:
        """Runs the entire enrichment pipeline on a text record."""
        entities = self.extract_entities(text)
        sentiment = self.analyze_sentiment(text)
        brand_safety = self.check_brand_safety(text)
        topic = self.classify_topic(text)
        
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
            "topic": topic,
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
        print(f"  Topic: {result['topic']}")
        print(f"  Safety: {result['brand_safety']}")
        print("-" * 50)
