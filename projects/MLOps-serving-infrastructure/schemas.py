"""
API Schema Definitions
Defines Pydantic request and response models for safe serialization.
"""

from typing import List
from pydantic import BaseModel, Field


class BidRequest(BaseModel):
    user_id: str = Field(..., examples=["usr_9812"])
    brand_context: str = Field(..., examples=["Nike shoes launch"])
    estimated_ctr: float = Field(..., gt=0.0, lt=1.0, examples=[0.045])
    campaign_id: str = Field(..., examples=["cmp_320"])
    user_features: List[float] = Field(..., min_length=3, max_length=3, examples=[[0.12, -0.4, 0.9]])


class BidResponse(BaseModel):
    bid_price: float = Field(..., examples=[2.14])
    should_bid: bool = Field(..., examples=[True])
    targeted_creative_id: str = Field(..., examples=["crt_102"])
    brand_safety_status: str = Field(..., examples=["SAFE"])


class DriftCheckRequest(BaseModel):
    baseline_features: List[float]
    production_features: List[float]


class DriftCheckResponse(BaseModel):
    ks_statistic: float
    p_value: float
    psi_metric: float
    drift_detected: bool
    status: str
