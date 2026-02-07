import os
import json
import logging
import joblib
import uvicorn
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from huggingface_hub import hf_hub_download
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration & Global State ---
app = FastAPI(title="Equilibrium Systemic Risk API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolve paths relative to this file's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "..", "frontend", "config.json")

MODEL_REPO = "Vansh180/Equilibrium-India-V1"
MODEL_FILENAME = "systemic_risk_model.keras"
SCALER_FILENAME = "scaler.pkl"
FEATURE_COLUMNS_FILENAME = "feature_columns.json"

# Global variables (loaded at startup)
config = {}
model = None
scaler = None
feature_columns = []

# --- Pydantic Models ---
class PredictionRequest(BaseModel):
    tickers: List[str]
    connectivity_scale: float = Field(default=1.0, gt=0, description="Scale factor for connectivity (must be > 0)")
    liquidity_buffer_scale: float = Field(default=1.0, gt=0, description="Scale factor for liquidity buffer (must be > 0)")

class CCPFundsOutput(BaseModel):
    initial_margin: float
    variation_margin_flow: float
    default_fund: float
    ccp_capital: float
    units: str = "indexed"
    vm_is_flow: bool = True

class PredictionResponse(BaseModel):
    predicted_next_systemic_risk: float
    latest_S_t: float
    latest_features: Dict[str, float]
    used_tickers: List[str]
    masked_tickers: List[str]
    end_date: str
    ccp_funds: CCPFundsOutput

# --- CCP Fund Computation ---

# Baseline constants (indexed units)
IM0 = 100
VM0 = 20
DF0 = 40
C0 = 10

# Reference values for normalization (typical observed ranges from training data)
# lambda_max typically ranges 1.5-3.5 for correlation matrices with CCP
# std_risk typically ranges 0.0-0.4 for normalized risk metrics
LAMBDA_REF = 2.5  # Reference lambda_max for normalization
STD_REF = 0.25    # Reference std_risk for normalization

def compute_ccp_funds(
    systemic: float,
    lambda_max: float,
    std_risk: float,
    connectivity_scale: float = 1.0,
    liquidity_buffer_scale: float = 1.0
) -> Dict[str, Any]:
    """
    Compute CCP fund requirements based on systemic risk and network metrics.
    
    Args:
        systemic: Predicted systemic risk score (0-1)
        lambda_max: Maximum eigenvalue of adjacency matrix
        std_risk: Standard deviation of risk across nodes
        connectivity_scale: Scenario override for connectivity (default 1.0)
        liquidity_buffer_scale: Scenario override for liquidity buffer (default 1.0)
    
    Returns:
        Dictionary with IM, VM, DF, CCP capital metrics
    """
    # Normalize lambda_max and std_risk to 0-1 range using reference values
    # Values above reference map to >1, below to <1
    lambda_norm = min(1.0, lambda_max / LAMBDA_REF) if LAMBDA_REF > 0 else 0.0
    std_norm = min(1.0, std_risk / STD_REF) if STD_REF > 0 else 0.0
    
    # Compute CCP stress from normalized values, then clamp to [0,1]
    # This ensures stress responds to changes rather than saturating
    ccp_stress = 0.4 * lambda_norm + 0.3 * std_norm + 0.3 * systemic
    ccp_stress = max(0.0, min(1.0, ccp_stress))
    
    # Compute 4 values based on stress
    im = IM0 * (1 + 1.5 * ccp_stress)
    vm = VM0 * (1 + 2.0 * systemic)  # VM is a flow based on systemic only
    df = DF0 * (1 + 2.0 * max(0, ccp_stress - 0.5))
    
    # CCP capital scales with stress: C_t = C_0 * (1 + k_C * CCPStress), clamped >= C_0
    k_C = 9.0  # Scaling factor for range 10-100
    ccp_capital = max(C0, C0 * (1 + k_C * ccp_stress))
    
    # Apply scenario overrides
    im *= connectivity_scale
    df *= connectivity_scale
    vm *= 1.0 / max(liquidity_buffer_scale, 1e-6)  # Tight liquidity => bigger VM
    
    return {
        "initial_margin": round(im, 4),
        "variation_margin_flow": round(vm, 4),
        "default_fund": round(df, 4),
        "ccp_capital": round(ccp_capital, 4),
        "units": "indexed",
        "vm_is_flow": True
    }

def _run_ccp_funds_self_checks():
    """Run self-checks to validate compute_ccp_funds logic."""
    # Check 1: If systemic increases, IM/VM should increase
    low_sys = compute_ccp_funds(0.2, 0.5, 0.3)
    high_sys = compute_ccp_funds(0.8, 0.5, 0.3)
    assert high_sys["initial_margin"] >= low_sys["initial_margin"], "IM should increase with systemic"
    assert high_sys["variation_margin_flow"] >= low_sys["variation_margin_flow"], "VM should increase with systemic"
    
    # Check 2: DF should increase only when ccp_stress > 0.5
    low_stress = compute_ccp_funds(0.1, 0.1, 0.1)  # ccp_stress = 0.4*0.1 + 0.3*0.1 + 0.3*0.1 = 0.1
    high_stress = compute_ccp_funds(0.8, 0.8, 0.8)  # ccp_stress = 0.4*0.8 + 0.3*0.8 + 0.3*0.8 = 0.8
    assert low_stress["default_fund"] == DF0, "DF should stay at baseline when ccp_stress <= 0.5"
    assert high_stress["default_fund"] > DF0, "DF should increase when ccp_stress > 0.5"
    
    # Check 3: Increasing connectivity_scale increases IM and DF
    base = compute_ccp_funds(0.5, 0.5, 0.5, connectivity_scale=1.0)
    scaled = compute_ccp_funds(0.5, 0.5, 0.5, connectivity_scale=1.5)
    assert scaled["initial_margin"] > base["initial_margin"], "IM should increase with connectivity_scale"
    assert scaled["default_fund"] > base["default_fund"], "DF should increase with connectivity_scale"
    
    logger.info("CCP funds self-checks passed.")

# --- Helper Functions ---

def load_config():
    """Load configuration from frontend/config.json."""
    global config
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r") as f:
                config = json.load(f)
            logger.info("Config loaded successfully.")
        else:
             # Fallback default config if file is missing (though unlikely in this setup)
            logger.warning(f"Config file not found at {CONFIG_PATH}. Using defaults.")
            config = {
                "tickers": ["HDFCBANK.NS", "KOTAKBANK.NS", "ICICIBANK.NS", "BAJFINANCE.NS", "BSE.NS", 
                            "TCS.NS", "INFY.NS", "RELIANCE.NS", "SBIN.NS", "ADANIENT.NS", 
                            "MRF.NS", "HINDUNILVR.NS", "TATASTEEL.NS", "AXISBANK.NS", "BHARTIARTL.NS"],
                "ccp_name": "CCP",
                "start": "2022-01-01",
                "ret_window": 20,
                "lookback": 20,
                "delta_ccp": 0.1
            }
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise

def setup_model_artifacts():
    """Download and load model, scaler, and feature columns."""
    global model, scaler, feature_columns
    try:
        # 1. Download Model
        logger.info(f"Downloading {MODEL_FILENAME} from {MODEL_REPO}...")
        model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME, repo_type="model")
        model = tf.keras.models.load_model(model_path) # type: ignore
        logger.info("Model loaded successfully.")

        # 2. Download Scaler
        logger.info(f"Downloading {SCALER_FILENAME} from {MODEL_REPO}...")
        scaler_path = hf_hub_download(repo_id=MODEL_REPO, filename=SCALER_FILENAME, repo_type="model")
        scaler = joblib.load(scaler_path)
        logger.info("Scaler loaded successfully.")

        # 3. Download/Set Feature Columns
        try:
            logger.info(f"Attempting to download {FEATURE_COLUMNS_FILENAME}...")
            cols_path = hf_hub_download(repo_id=MODEL_REPO, filename=FEATURE_COLUMNS_FILENAME, repo_type="model")
            with open(cols_path, "r") as f:
                feature_columns = json.load(f)
            logger.info(f"Feature columns loaded: {feature_columns}")
        except Exception as e:
            logger.warning(f"Could not load feature_columns.json ({e}). Using default columns.")
            feature_columns = ["lambda_max", "mean_risk", "max_risk", "std_risk", "S_lag1", "S_lag5"]

    except Exception as e:
        logger.critical(f"Failed to setup model artifacts: {e}")
        raise RuntimeError("Model initialization failed.")

# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    load_config()
    setup_model_artifacts()
    _run_ccp_funds_self_checks()  # Validate CCP funds logic

# --- Core Logic ---

def compute_rolling_metrics(returns, window=20):
    """Compute rolling volatility and rolling max drawdown proxy."""
    # Rolling Volatility
    rolling_std = returns.rolling(window=window).std()
    
    # Rolling Drawdown Proxy (simple)
    # Using rolling max of price would be better, but we have returns. 
    # Let's reconstruct a cumulative return series for drawdown calculation roughly or use return based proxy.
    # Standard practice with just returns:
    # We need prices for standard drawdown. Let's assume we can use cumulative sum of log returns as log-price.
    
    log_prices = returns.cumsum()
    rolling_max = log_prices.rolling(window=window, min_periods=1).max()
    drawdown = (log_prices - rolling_max) # This is log-drawdown, roughly % drawdown
    # Invert so it's a positive risk metric? Drawdown is negative.
    # Risk is usually magnitude. Let's take abs of drawdown (distance from peak).
    rolling_drawdown = drawdown.abs()
    
    return rolling_std, rolling_drawdown

def compute_features_for_subset(tickers_subset: List[str]):
    """
    Main computational pipeline.
    1. Fetch data for ALL config tickers.
    2. Compute market-wide metrics (Adjacency, Eigenvector).
    3. Apply masking: Set risk of unselected tickers to 0.
    4. Compute Systemic Risk Payoff S_t.
    5. Construct feature lags.
    """
    all_tickers = config["tickers"]
    start_date = config["start"]
    ccp_name = config["ccp_name"]
    delta_ccp = config["delta_ccp"]
    lookback = config["lookback"]
    
    # Identify indices
    try:
        # Create a boolean mask for selected tickers
        # We need to maintain the order of 'all_tickers' for matrix operations
        mask_vector = np.array([1.0 if t in tickers_subset else 0.0 for t in all_tickers])
    except Exception as e:
         raise HTTPException(status_code=400, detail=f"Error processing ticker subset: {e}")

    # 1. Fetch Data
    # We fetch data until today.
    # yfinance auto_adjust=True
    logger.info("Fetching market data...")
    try:
        raw_data = yf.download(all_tickers, start=start_date, progress=False, auto_adjust=True, threads=False)
        if raw_data is None or raw_data.empty:
            raise HTTPException(status_code=500, detail="No data returned from yfinance.")
        data = raw_data['Close']
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch data from yfinance: {e}")

    if data is None or data.empty:
        raise HTTPException(status_code=500, detail="No Close price data returned from yfinance.")
    
    # Handle single ticker case (though unlikely given list) causing Series instead of DataFrame
    if isinstance(data, pd.Series):
        data = data.to_frame()

    # Reorder columns to match all_tickers list exactly (yfinance might sort them)
    # Filter only columns that exist (in case some tickers failed)
    existing_tickers = [t for t in all_tickers if t in data.columns]
    data = data[existing_tickers]
    
    # Recalculate mask based on existing tickers (if any were dropped by yfinance)
    mask_vector = np.array([1.0 if t in tickers_subset and t in existing_tickers else 0.0 for t in existing_tickers])
    N = len(existing_tickers)

    # 2. Log Returns
    returns = np.log(data / data.shift(1)).dropna()

    # 3. Rolling Calculations
    # We need to iterate over time to compute A_t, v_t, r_t, S_t
    # Correlation window = 20
    window = 20
    
    # Lists to store time-series of metrics
    s_t_series = []
    lambda_max_series = []
    
    # Store other risk metrics for feature creation
    mean_risk_series = []
    max_risk_series = []
    std_risk_series = []

    # Rolling Volatility & Drawdown (Vectorized)
    rolling_std, rolling_dd = compute_rolling_metrics(returns, window)
    
    # To compute min-max normalization, we need expanding window/historic min-max.
    # Let's simplify and do expanding window from start of data.
    
    # Combined Risk Metric: Volatility + Drawdown (Equal weight?)
    # "rolling vol + rolling drawdown proxy" - Sum?
    raw_risk = rolling_std + rolling_dd
    
    # Normalize [0,1] expanding
    # Note: expanding().min() / max() might be slow in loop, let's vectorise
    expanding_min = raw_risk.expanding().min()
    expanding_max = raw_risk.expanding().max()
    
    # Avoid div by zero
    denom = expanding_max - expanding_min
    denom = denom.replace(0, 1.0) # Handle constant case
    
    norm_risk = (raw_risk - expanding_min) / denom
    
    # We can only compute calculating from `window` onwards
    # And we need enough history for lags (lag 5 is max)
    # We need to return exactly `lookback` (20) days of features.
    # BUT, to compute S_lag5 for the *first* of those 20 days, we need S from 5 days before that.
    # So we need to compute S_t for range: [end - lookback - 5, end]
    
    required_history_len = lookback + 5 
    
    # Ensure we have enough data
    if len(returns) < window + required_history_len:
         raise HTTPException(status_code=400, detail=f"Insufficient data history. Need at least {window + required_history_len} days.")
         
    # Slice the relevant period for iteration
    # We usually want the "latest" prediction, so we process the tail.
    # Let's process the last (required_history_len) days.
    
    process_indices = returns.index[-required_history_len:]
    
    # Pre-compute Rolling Correlations for efficiency?
    # rolling(window).corr() returns a MultiIndex series.
    # It might be heavy to do for all history. Let's do loop for just the needed days.
    
    # History of S metrics to build lags
    history_S = []
    
    for date in process_indices:
        # 1. Get correlation matrix for window ending at 'date'
        # Data slice: date-window+1 to date
        # returns.loc[:date].tail(window)
        window_returns = returns.loc[:date].tail(window)
        
        if len(window_returns) < window:
             # Should not happen given logic above
             s_t_series.append(0)
             continue
             
        # Correlation
        corr_mat = window_returns.corr().values
        # Fill NaNs (if constant price) with 0
        corr_mat = np.nan_to_num(corr_mat)
        
        # 2. Adjacency Matrix A
        # Off-diagonal = max(0, corr)
        A = np.maximum(0, corr_mat)
        np.fill_diagonal(A, 0)
        
        # 3. Add CCP
        # A is N x N. New A is (N+1) x (N+1)
        # Append column and row
        # Column N: 0.1s
        # Row N: 0.1s
        # A[N,N] = 0
        
        A_ext = np.zeros((N+1, N+1))
        
        # Copy bank-bank block
        A_ext[:N, :N] = A
        
        # Add CCP edges
        A_ext[:N, N] = delta_ccp # Bank -> CCP
        A_ext[N, :N] = delta_ccp # CCP -> Bank
        
        # 4. Compute Principal Eigenvector & Lambda Max
        # Power iteration or linalg.eigh
        # Since A is symmetric (corr is symmetric), eigh is good.
        eigvals, eigvecs = np.linalg.eigh(A_ext)
        
        # Max eigenvalue and vector
        lambda_max = eigvals[-1]
        v_t = eigvecs[:, -1]
        
        # Ensure v_t is positive (Perron-Frobenius for non-negative matrices implies there's a non-negative eigenvector)
        # Sometimes solver flips sign.
        if np.sum(v_t) < 0:
            v_t = -v_t
            
        # 5. Node Risks r_t
        # Get risk for this date
        # norm_risk is a DataFrame, get row, convert to array
        r_t_banks = norm_risk.loc[date].values
        
        # Apply MASKING
        # Set unselected tickers to 0
        r_t_banks_masked = r_t_banks * mask_vector
        
        # CCP Risk = 0
        r_t_ext = np.append(r_t_banks_masked, 0.0)
        
        # 6. Payoff S_t = r_t^T * A * v_t
        # Dot product
        # A * v
        Av = np.dot(A_ext, v_t)
        # r * Av
        S_t = np.dot(r_t_ext, Av)
        
        # 7. Statistics for Features
        # Compute stats on the MASKED banks risk (excluding CCP)
        # "mean_risk, max_risk, std_risk"
        # Using masked values might skew mean to 0 if many are masked. 
        # But this reflects the "effective" system state if unselected are removed.
        # Let's use the masked vectors.
        # Note: If we really masked checks (set to 0), maybe we should exclude them from mean/std?
        # But for fixed feature vector size, usually we just compute on the vector.
        
        mean_r = np.mean(r_t_banks_masked)
        max_r = np.max(r_t_banks_masked)
        std_r = np.std(r_t_banks_masked)
        
        # Store
        history_S.append(S_t)
        lambda_max_series.append(lambda_max)
        mean_risk_series.append(mean_r)
        max_risk_series.append(max_r)
        std_risk_series.append(std_r)

    # Convert history to DataFrame to build features
    feature_df = pd.DataFrame({
        "lambda_max": lambda_max_series,
        "mean_risk": mean_risk_series,
        "max_risk": max_risk_series,
        "std_risk": std_risk_series,
        "S_t": history_S
    }, index=process_indices)
    
    # Create Lags
    feature_df["S_lag1"] = feature_df["S_t"].shift(1)
    feature_df["S_lag5"] = feature_df["S_t"].shift(5)
    
    # Drop NaNs created by shifting
    # We calculated `lookback + 5` days.
    # Shifting by 5 loses first 5.
    feature_df = feature_df.dropna()
    
    # Select last `lookback` (20) rows
    feature_df = feature_df.tail(lookback)
    
    if len(feature_df) < lookback:
         logger.error(f"Not enough data after lag creation. Have {len(feature_df)}, need {lookback}")
         raise HTTPException(status_code=400, detail="Insufficient data for feature window.")
         
    # Select feature columns in correct order
    # Ensure columns match model expectation
    final_features = feature_df[feature_columns]
    
    return final_features, existing_tickers

# --- Endpoint ---
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if not model or not scaler:
        raise HTTPException(status_code=503, detail="Model not loaded.")
        
    # Input validation
    tickers_subset = request.tickers
    if not tickers_subset:
        raise HTTPException(status_code=400, detail="Tickers list cannot be empty.")
        
    valid_tickers = set(config["tickers"])
    invalid = [t for t in tickers_subset if t not in valid_tickers]
    if invalid:
        raise HTTPException(status_code=400, detail=f"Invalid tickers: {invalid}. Must be in config.")

    # Compute Features
    try:
        features_df, available_tickers = compute_features_for_subset(tickers_subset)
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Computation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
        
    # Scale Features
    X = features_df.values
    try:
        X_scaled = scaler.transform(X)
    except Exception as e:
        logger.error(f"Scaling error: {e}")
        raise HTTPException(status_code=500, detail=f"Scaling failed: {e}")
        
    # Reshape for LSTM/GRU: (1, 20, 6)
    # features_df should have 20 rows
    X_reshaped = X_scaled.reshape(1, config["lookback"], len(feature_columns))
    
    # Predict
    try:
        prediction = model.predict(X_reshaped)
        risk_score = float(prediction[0][0])
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")
        
    # Latest Data
    latest_row = features_df.iloc[-1]
    last_date = features_df.index[-1].strftime("%Y-%m-%d")
    
    # Extract metrics for CCP funds computation
    # lambda_max and std_risk from latest features
    lambda_max_val = float(latest_row.get("lambda_max", 0.0))
    std_risk_val = float(latest_row.get("std_risk", 0.0))
    
    # Compute CCP funds
    ccp_funds = compute_ccp_funds(
        systemic=risk_score,
        lambda_max=lambda_max_val,
        std_risk=std_risk_val,
        connectivity_scale=request.connectivity_scale,
        liquidity_buffer_scale=request.liquidity_buffer_scale
    )
    
    return {
        "predicted_next_systemic_risk": risk_score,
        "latest_S_t": latest_row.get("S_t", 0.0) if "S_t" in latest_row else 0.0,
        "latest_features": latest_row.to_dict(),
        "used_tickers": available_tickers,
        "masked_tickers": [t for t in config["tickers"] if t not in tickers_subset],
        "end_date": last_date,
        "ccp_funds": ccp_funds
    }

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}
