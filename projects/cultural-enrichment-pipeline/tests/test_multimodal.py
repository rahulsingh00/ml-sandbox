"""
Unit tests for multimodal_enrichment.py using pytest.
Mocks CLIP model encoder layers for fast execution.
"""

from unittest.mock import patch, MagicMock
import numpy as np
import pytest
from multimodal_enrichment import MultimodalEnrichmentPipeline


@pytest.fixture
def multimodal_pipeline():
    # Force HAS_CLIP_DEPENDENCIES to True for testing the CLIP code path
    import multimodal_enrichment
    multimodal_enrichment.HAS_CLIP_DEPENDENCIES = True
    
    with patch("multimodal_enrichment.CLIPModel") as mock_model, \
         patch("multimodal_enrichment.CLIPProcessor") as mock_processor:
        
        # Mocking the initialization
        mock_model.from_pretrained.return_value = MagicMock()
        mock_processor.from_pretrained.return_value = MagicMock()
        
        pipeline = MultimodalEnrichmentPipeline()
        pipeline.has_clip = True
        pipeline.model = MagicMock()
        pipeline.processor = MagicMock()
        pipeline.device = "cpu"
        
        yield pipeline


def test_extract_image_embedding_clip(multimodal_pipeline):
    # Setup mock return for CLIP image features
    # Return a 512-dim PyTorch tensor
    import torch
    mock_features = torch.randn(1, 512)
    # L2 normalized mock
    mock_features /= mock_features.norm(dim=-1, keepdim=True)
    
    multimodal_pipeline.model.get_image_features.return_value = mock_features
    multimodal_pipeline.processor.return_value = {"pixel_values": torch.randn(1, 3, 224, 224)}
    
    emb = multimodal_pipeline.extract_image_embedding("mock_path.jpg")
    
    # Assert return properties
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (512,)
    # Verify L2 normalized (norm is close to 1)
    assert np.allclose(np.linalg.norm(emb), 1.0, atol=1e-5)


def test_extract_text_embeddings_clip(multimodal_pipeline):
    import torch
    labels = ["safe", "unsafe"]
    mock_features = torch.randn(2, 512)
    mock_features /= mock_features.norm(dim=-1, keepdim=True)
    
    multimodal_pipeline.model.get_text_features.return_value = mock_features
    multimodal_pipeline.processor.return_value = {"input_ids": torch.randint(0, 100, (2, 10))}
    
    embs = multimodal_pipeline.extract_text_embeddings(labels)
    
    assert len(embs) == 2
    assert "safe" in embs
    assert "unsafe" in embs
    assert embs["safe"].shape == (512,)
    assert np.allclose(np.linalg.norm(embs["safe"]), 1.0, atol=1e-5)


def test_evaluate_image_safety(multimodal_pipeline):
    # Mock image and text features to get a controlled dot product (cosine similarity)
    # We want "safe environment" to have highest similarity
    image_emb = np.zeros(512)
    image_emb[0] = 1.0 # One-hot
    
    text_embs = {
        "safe environment": np.zeros(512),
        "hate speech banner": np.zeros(512),
        "physical violence": np.zeros(512),
        "weapon or firearm": np.zeros(512)
    }
    text_embs["safe environment"][0] = 0.9
    text_embs["hate speech banner"][0] = 0.2
    
    with patch.object(multimodal_pipeline, "extract_image_embedding", return_value=image_emb), \
         patch.object(multimodal_pipeline, "extract_text_embeddings", return_value=text_embs):
        
        report = multimodal_pipeline.evaluate_image_safety("dummy.jpg")
        
        assert report["is_safe"]
        assert report["primary_visual_concept"] == "safe environment"
        assert report["similarity_scores"]["safe environment"] == 0.9


def test_evaluate_video_safety(multimodal_pipeline):
    import torch
    # Mock CLIP image features return
    mock_features = torch.zeros(1, 512)
    mock_features[0, 1] = 1.0  # Set index 1 to 1.0 to match the frame_emb logic
    
    multimodal_pipeline.model.get_image_features.return_value = mock_features
    multimodal_pipeline.processor.return_value = {"pixel_values": torch.randn(1, 3, 224, 224)}

    text_embs = {
        "safe environment": np.zeros(512),
        "hate speech banner": np.zeros(512),
        "physical violence": np.zeros(512),
        "weapon or firearm": np.zeros(512)
    }
    # Let "weapon or firearm" have highest similarity
    text_embs["weapon or firearm"][1] = 0.95
    
    # We mock cv2.VideoCapture to simulate reading 2 frames
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    # First two reads return True, frame; third returns False, None (EOF)
    mock_cap.read.side_effect = [
        (True, np.zeros((100, 100, 3), dtype=np.uint8)),
        (True, np.zeros((100, 100, 3), dtype=np.uint8)),
        (False, None)
    ]
    mock_cap.get.return_value = 10.0 # FPS
    
    with patch("cv2.VideoCapture", return_value=mock_cap), \
         patch.object(multimodal_pipeline, "extract_text_embeddings", return_value=text_embs):
        
        # Execute safety check
        report = multimodal_pipeline.evaluate_video_safety("dummy_video.mp4", fps=5.0)
        
        assert not report["is_safe"]
        assert report["primary_visual_concept"] == "weapon or firearm"
        assert report["similarity_scores"]["weapon or firearm"] == 0.95

