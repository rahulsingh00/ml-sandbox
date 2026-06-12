"""
FastAPI Server Entrypoint
Implements async endpoints for real-time bid scoring, serving,
and drift evaluation checks.
"""

from fastapi import FastAPI, HTTPException
from schemas import BidRequest, BidResponse, DriftCheckRequest, DriftCheckResponse
from drift_detector import PopulationDriftMonitor

app = FastAPI(
    title="AdTech ML Serving Infrastructure",
    description="Asynchronous serving layer for real-time bidding decisions and drift monitoring.",
    version="1.0.0"
)

# Mock models & heuristics local parameters
MARKET_INFLECTION_PRICE = 2.0
COMPETITIVENESS_SLOPE = 1.5


@app.get("/health")
async def health_check():
    """Simple API health diagnostic check endpoint."""
    return {"status": "HEALTHY", "version": "1.0.0"}


@app.post("/bid", response_model=BidResponse)
async def score_and_bid(request: BidRequest):
    """
    Computes utility-maximizing bid and targets creative.
    Applies real-time brand safety logic.
    """
    try:
        # Check brand safety
        brand_safety_status = "SAFE"
        unsafe_keywords = ["violence", "hate speech", "illegal", "terrorism"]
        if any(w in request.brand_context.lower() for w in unsafe_keywords):
            brand_safety_status = "UNSAFE"
            return BidResponse(
                bid_price=0.0,
                should_bid=False,
                targeted_creative_id="none",
                brand_safety_status=brand_safety_status
            )
            
        # Optimize bid price: b* = (value - bid) * P_win(bid)
        # Assuming Value of impression = estimated_ctr * 50
        val = request.estimated_ctr * 50.0
        
        # Simple analytic approximation for testing speed
        # If value is low, bid low; if high, shade towards market inflection
        bid_price = min(val * 0.6, MARKET_INFLECTION_PRICE * 1.2)
        
        # Determine creative targeting using feature vectors
        # If user feature index 0 is positive, target Creative A
        targeted_creative = "crt_101" if request.user_features[0] > 0 else "crt_102"
        
        return BidResponse(
            bid_price=round(bid_price, 2),
            should_bid=bid_price > 0.1,
            targeted_creative_id=targeted_creative,
            brand_safety_status=brand_safety_status
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/check-drift", response_model=DriftCheckResponse)
async def verify_data_drift(request: DriftCheckRequest):
    """Calculates KS-test statistical differences and PSI metric values."""
    try:
        monitor = PopulationDriftMonitor()
        ks_stat, p_val = monitor.kolmogorov_smirnov_test(
            request.baseline_features, request.production_features
        )
        psi_val = monitor.population_stability_index(
            request.baseline_features, request.production_features
        )
        
        drift_detected = p_val < 0.05 or psi_val >= 0.25
        status_msg = "RETRAIN_MODEL" if drift_detected else "STABLE"
        
        return DriftCheckResponse(
            ks_statistic=round(ks_stat, 4),
            p_value=float(p_val),
            psi_metric=round(psi_val, 4),
            drift_detected=drift_detected,
            status=status_msg
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
