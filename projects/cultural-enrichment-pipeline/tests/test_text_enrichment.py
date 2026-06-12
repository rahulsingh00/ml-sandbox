"""
Unit tests for text_enrichment.py using pytest.
Mocks Hugging Face pipelines for fast execution.
"""

from unittest.mock import patch, MagicMock
import pytest
from text_enrichment import TextEnrichmentPipeline


@pytest.fixture
def pipeline_obj():
    # Reset HAS_TRANSFORMERS locally to test transformers execution path
    import text_enrichment
    text_enrichment.HAS_TRANSFORMERS = True
    obj = TextEnrichmentPipeline()
    return obj


@patch("text_enrichment.pipeline")
def test_extract_entities_hf(mock_pipeline_func, pipeline_obj):
    # Mock NER pipeline
    mock_ner = MagicMock()
    mock_ner.return_value = [
        {"entity_group": "ORG", "score": 0.98, "word": "Microsoft", "start": 0, "end": 9},
        {"entity_group": "PER", "score": 0.95, "word": "Satya Nadella", "start": 15, "end": 28}
    ]
    
    mock_pipeline_func.return_value = mock_ner
    
    # Force initialize the pipeline object's mock
    pipeline_obj.ner_pipeline = mock_ner
    
    text = "Microsoft CEO Satya Nadella announced new updates #Tech."
    entities = pipeline_obj.extract_entities(text)
    
    # Assert NER extraction
    assert any(e["entity"] == "Microsoft" and e["category"] == "BRAND" for e in entities)
    assert any(e["entity"] == "Satya Nadella" and e["category"] == "PERSON" for e in entities)
    
    # Assert regex hashtag fallback
    assert any(e["entity"] == "#Tech" and e["category"] == "HASHTAG" for e in entities)


@patch("text_enrichment.pipeline")
def test_analyze_sentiment_hf(mock_pipeline_func, pipeline_obj):
    # Mock Sentiment pipeline
    mock_sent = MagicMock()
    mock_sent.return_value = [{"label": "positive", "score": 0.92}]
    
    mock_pipeline_func.return_value = mock_sent
    pipeline_obj.sentiment_pipeline = mock_sent
    
    result = pipeline_obj.analyze_sentiment("I absolutely love this!")
    assert result["sentiment"] == "POSITIVE"
    assert result["score"] == 0.92


@patch("text_enrichment.pipeline")
def test_classify_stance_hf(mock_pipeline_func, pipeline_obj):
    # Mock zero-shot classifier for stance
    mock_classifier = MagicMock()
    mock_classifier.return_value = {
        "labels": ["against", "in favor of", "neutral toward"],
        "scores": [0.85, 0.10, 0.05]
    }
    
    mock_pipeline_func.return_value = mock_classifier
    pipeline_obj.zero_shot_pipeline = mock_classifier
    
    stance = pipeline_obj.classify_stance("We need to boycott Sightly.", "Sightly")
    assert stance == "AGAINST"


@patch("text_enrichment.pipeline")
def test_classify_topic_hf(mock_pipeline_func, pipeline_obj):
    # Mock zero-shot classifier for topic
    mock_classifier = MagicMock()
    mock_classifier.return_value = {
        "labels": ["Tech", "Finance", "Sports"],
        "scores": [0.90, 0.08, 0.02]
    }
    
    mock_pipeline_func.return_value = mock_classifier
    pipeline_obj.zero_shot_pipeline = mock_classifier
    
    topic_result = pipeline_obj.classify_topic("Running computations on high-performance GPUs.")
    assert topic_result["primary_topic"] == "Tech"
    assert topic_result["topic_distribution"]["Tech"] == 0.90


@patch("text_enrichment.pipeline")
def test_check_brand_safety_hf(mock_pipeline_func, pipeline_obj):
    # Mock toxicity pipeline
    mock_toxic = MagicMock()
    mock_toxic.return_value = [
        [{"label": "toxic", "score": 0.88}, {"label": "severe_toxic", "score": 0.12}]
    ]
    
    mock_pipeline_func.return_value = mock_toxic
    pipeline_obj.safety_pipeline = mock_toxic
    
    safety = pipeline_obj.check_brand_safety("Some random text with potential toxicity.")
    assert not safety["is_safe"]
    assert safety["risk_score"] == 0.88
    assert "model_flagged_toxicity" in safety["flagged_categories"]


def test_fallback_heuristics():
    # Test execution when transformers are disabled
    import text_enrichment
    text_enrichment.HAS_TRANSFORMERS = False
    
    pipeline_fallback = TextEnrichmentPipeline()
    
    # Test sentiment fallback
    sent = pipeline_fallback.analyze_sentiment("This is a great, awesome day.")
    assert sent["sentiment"] == "POSITIVE"
    assert sent["score"] > 0.0
    
    # Test stance fallback
    stance = pipeline_fallback.classify_stance("We must boycott Nike.", "Nike")
    assert stance == "AGAINST"
    
    # Test brand safety fallback
    safety = pipeline_fallback.check_brand_safety("This contains drugs and violence.")
    assert not safety["is_safe"]
    assert "violence" in safety["flagged_categories"]
