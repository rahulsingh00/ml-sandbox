"""
CTR Regression Predictor Module
Implements Gradient Boosting Regressor for continuous-value CTR prediction.
Demonstrates training data extraction with a Snowflake integration stub and local fallback.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    import snowflake.connector
    HAS_SNOWFLAKE = True
except ImportError:
    HAS_SNOWFLAKE = False


class CTRPredictor:
    """Predicts continuous Click-Through Rate (CTR) for ad bids using Gradient Boosting Regressor."""

    def __init__(self):
        self.model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
        self.is_trained = False
        self.feature_columns = None

    def generate_synthetic_data(self, num_samples: int = 1000) -> pd.DataFrame:
        """
        Generates synthetic ad performance data with realistic CTR distributions
        and non-linear feature interactions.
        """
        rng = np.random.default_rng(42)
        
        # Features
        hour_of_day = rng.integers(0, 24, size=num_samples)
        day_of_week = rng.integers(0, 7, size=num_samples)
        device_type = rng.choice(["mobile", "desktop", "tablet", "connected_tv"], size=num_samples)
        placement = rng.choice(["banner", "interstitial", "rewarded_video", "native"], size=num_samples)
        historical_ctr = rng.normal(loc=0.015, scale=0.005, size=num_samples)
        historical_ctr = np.clip(historical_ctr, 0.001, 0.10)
        bid_floor = rng.exponential(scale=1.5, size=num_samples) + 0.1
        bid_floor = np.clip(bid_floor, 0.1, 10.0)
        
        # Build DataFrame
        df = pd.DataFrame({
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week,
            "device_type": device_type,
            "placement": placement,
            "historical_ctr": historical_ctr,
            "bid_floor": bid_floor
        })
        
        # Generate non-linear target actual_ctr with noise
        # Base CTR is historical_ctr
        ctr = df["historical_ctr"].copy()
        
        # Device effects
        ctr += df["device_type"].map({"mobile": 0.006, "desktop": 0.002, "tablet": 0.001, "connected_tv": -0.002})
        
        # Placement effects
        ctr += df["placement"].map({"banner": -0.005, "interstitial": 0.008, "rewarded_video": 0.015, "native": 0.003})
        
        # Time-of-day peak effect (evening peak hours)
        peak_mask = (df["hour_of_day"] >= 18) & (df["hour_of_day"] <= 22)
        ctr += peak_mask.astype(float) * 0.004
        
        # Bid floor effect (higher floor implies premium inventory with higher CTR)
        ctr += np.log1p(df["bid_floor"]) * 0.002
        
        # Add normal noise
        noise = rng.normal(loc=0.0, scale=0.002, size=num_samples)
        ctr += noise
        
        # Clip CTR in realistic bounds
        df["actual_ctr"] = np.clip(ctr, 0.0001, 0.20)
        return df

    def load_training_data_from_snowflake(self, connection_params: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Extracts training data from Snowflake tables.
        Demonstrates the production Snowflake integration pattern.
        If snowflake-connector-python is available and parameters are provided, executes query.
        Otherwise, falls back to generating synthetic training data locally.
        """
        if HAS_SNOWFLAKE and connection_params:
            try:
                # Establish connection
                conn = snowflake.connector.connect(**connection_params)
                cursor = conn.cursor()
                query = """
                    SELECT HOUR_OF_DAY, DAY_OF_WEEK, DEVICE_TYPE, PLACEMENT, HISTORICAL_CTR, BID_FLOOR, ACTUAL_CTR
                    FROM AD_PERFORMANCE_METRICS
                    SAMPLE (10) -- 10% Bernoulli sampling for data scale handling
                """
                cursor.execute(query)
                df = cursor.fetch_pandas_all()
                cursor.close()
                conn.close()
                print("Successfully loaded training data from Snowflake.")
                # Ensure column names are lowercase
                df.columns = [col.lower() for col in df.columns]
                return df
            except Exception as e:
                print(f"Warning: Snowflake query execution failed ({e}). Falling back to local data generation.")
        else:
            print("Note: Snowflake connector not found or connection parameters missing. Running in local sandbox mode.")
            
        return self.generate_synthetic_data(num_samples=1500)

    def preprocess_data(self, df: pd.DataFrame, is_training: bool = True) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        One-hot encodes categorical columns and aligns columns for training/prediction.
        """
        features_df = df[["hour_of_day", "day_of_week", "historical_ctr", "bid_floor"]].copy()
        
        # One-hot encode device_type and placement
        categorical_cols = ["device_type", "placement"]
        for col in categorical_cols:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col)
                features_df = pd.concat([features_df, dummies], axis=1)
                
        if is_training:
            self.feature_columns = features_df.columns.tolist()
            y = df["actual_ctr"].values
            return features_df, y
        else:
            # Reindex columns to match training schema, filling missing dummies with 0
            features_df = features_df.reindex(columns=self.feature_columns, fill_value=0)
            return features_df, None

    def fit(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Preprocesses and splits data, fits the Gradient Boosting model, and computes validation metrics.
        """
        X, y = self.preprocess_data(df, is_training=True)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Fit model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Predict on test
        y_pred = self.model.predict(X_test)
        
        # Calculate regression metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return {
            "test_mse": float(mse),
            "test_mae": float(mae),
            "test_r2": float(r2)
        }

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predicts continuous CTR values for inference samples."""
        if not self.is_trained:
            raise ValueError("CTRPredictor has not been fit/trained yet.")
        X, _ = self.preprocess_data(df, is_training=False)
        return self.model.predict(X)


if __name__ == "__main__":
    print("=== CTR REGRESSION PREDICTOR TEST ===\n")
    
    predictor = CTRPredictor()
    
    # Load data (falls back to synthetic)
    df_data = predictor.load_training_data_from_snowflake()
    print(f"Data shape: {df_data.shape}")
    print("Sample records:")
    print(df_data[["device_type", "placement", "historical_ctr", "actual_ctr"]].head(3))
    print()
    
    # Train the regressor
    print("Training Gradient Boosting Regressor...")
    metrics = predictor.fit(df_data)
    
    print("Validation Metrics:")
    print(f"  Mean Squared Error (MSE): {metrics['test_mse']:.6f}")
    print(f"  Mean Absolute Error (MAE): {metrics['test_mae']:.6f}")
    print(f"  R² Score: {metrics['test_r2']:.4f}")
    print()
    
    # Perform a sample prediction
    sample_request = pd.DataFrame({
        "hour_of_day": [20],
        "day_of_week": [4],
        "device_type": ["mobile"],
        "placement": ["rewarded_video"],
        "historical_ctr": [0.025],
        "bid_floor": [2.5]
    })
    
    pred_ctr = predictor.predict(sample_request)[0]
    print("Sample Inference:")
    print(f"  Input Features: Mobile, Rewarded Video, hour=20, hist_ctr=2.5%")
    print(f"  Predicted CTR: {pred_ctr:.4f} ({pred_ctr * 100:.2f}%)")
