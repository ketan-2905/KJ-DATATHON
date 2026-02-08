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
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- LLM Client Setup ---
llm_client = OpenAI(
    base_url="https://api.featherless.ai/v1",
    api_key=os.getenv("FEATHERLESS_API_KEY")
)

def analyze_cascade_with_llm(
    nodes: List[str],
    shocked_node: str,
    shock_magnitude: float,
    connectivity_strength: float,
    liquidity_buffer: float,
    systemic_risk: float,
    congestion_level: str
) -> Dict[str, Any]:
    """
    Use LLM to analyze cascade with HARD BOUNDARIES.
    Cascade triggers deterministically based on thresholds:
    - Shock magnitude > 0.6 (60%)
    - Connectivity strength > 1.3
    - Liquidity buffer < 0.7
    - Systemic risk > 0.5 (50%)
    """
    
    # HARD BOUNDARY RULES - cascade triggers if ANY of these:
    trigger_shock = shock_magnitude > 0.6
    trigger_connectivity = connectivity_strength > 1.3
    trigger_liquidity = liquidity_buffer < 0.7
    trigger_systemic = systemic_risk > 0.5
    trigger_congestion = congestion_level == "high"
    
    should_cascade = (trigger_shock and (trigger_connectivity or trigger_liquidity)) or \
                     (trigger_systemic and trigger_congestion) or \
                     (shock_magnitude > 0.8)  # Very high shock always cascades
    
    if not should_cascade:
        return {
            "status": "stable",
            "failed_nodes": [],
            "failed_edges": [],
            "failure_ratio": 0.0,
            "cascade_depth": 0,
            "analysis": "Network is stable. No cascade triggered."
        }
    
    # Use LLM to determine WHICH nodes fail given the cascade is triggered
    prompt = f"""You are a financial network cascade analyzer. Analyze this network stress scenario.

NETWORK NODES: {nodes}
SHOCKED NODE: {shocked_node}
SHOCK MAGNITUDE: {shock_magnitude * 100:.0f}%
CONNECTIVITY STRENGTH: {connectivity_strength}
LIQUIDITY BUFFER LEVEL: {liquidity_buffer}
SYSTEMIC RISK SCORE: {systemic_risk * 100:.0f}%
CONGESTION LEVEL: {congestion_level}

CASCADE HAS BEEN TRIGGERED. Determine which nodes fail based on these rules:
1. The shocked node ALWAYS fails first
2. If shock > 70%, nodes most connected to the shocked node fail next
3. If connectivity > 1.5, failures spread to 50% of remaining nodes
4. If liquidity < 0.5, all nodes fail
5. Otherwise, 1-3 additional nodes fail proportional to shock magnitude

Respond with ONLY valid JSON (no markdown, no extra text):
{{"failed_nodes": ["NODE1.NS", "NODE2.NS"], "cascade_depth": 2, "analysis": "brief explanation"}}

IMPORTANT: Only include nodes from the NETWORK NODES list. The shocked node must be in failed_nodes."""

    try:
        response = llm_client.chat.completions.create(
            model="moonshotai/Kimi-K2-Instruct",
            messages=[
                {"role": "system", "content": "You are a financial risk analyst. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low temperature for deterministic output
            max_tokens=500
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Clean up response - remove markdown code blocks if present
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
        result_text = result_text.strip()
        
        result = json.loads(result_text)
        failed_nodes = result.get("failed_nodes", [shocked_node])
        
        # Ensure shocked node is in failed list
        if shocked_node not in failed_nodes:
            failed_nodes.insert(0, shocked_node)
        
        # Filter to only valid nodes
        failed_nodes = [n for n in failed_nodes if n in nodes]
        
        # Generate failed edges (edges connected to failed nodes)
        failed_edges = []
        for fn in failed_nodes:
            for other in nodes:
                if other != fn and other not in failed_nodes:
                    failed_edges.append({"source": fn, "target": other})
        
        return {
            "status": "cascade",
            "failed_nodes": failed_nodes,
            "failed_edges": failed_edges,
            "failure_ratio": len(failed_nodes) / len(nodes) if nodes else 0,
            "cascade_depth": result.get("cascade_depth", 1),
            "analysis": result.get("analysis", "Cascade triggered due to network stress.")
        }
        
    except Exception as e:
        logger.error(f"LLM cascade analysis failed: {e}")
        # Fallback: deterministic cascade based on rules
        failed_nodes = [shocked_node]
        
        if shock_magnitude > 0.8:
            # Add ~50% of nodes
            additional = int(len(nodes) * 0.5)
            for n in nodes:
                if n != shocked_node and len(failed_nodes) < additional + 1:
                    failed_nodes.append(n)
        elif shock_magnitude > 0.6:
            # Add 1-2 nodes
            for n in nodes[:2]:
                if n != shocked_node:
                    failed_nodes.append(n)
        
        failed_edges = []
        for fn in failed_nodes:
            for other in nodes:
                if other != fn and other not in failed_nodes:
                    failed_edges.append({"source": fn, "target": other})
        
        return {
            "status": "cascade",
            "failed_nodes": failed_nodes,
            "failed_edges": failed_edges,
            "failure_ratio": len(failed_nodes) / len(nodes) if nodes else 0,
            "cascade_depth": 1,
            "analysis": f"Cascade triggered (fallback mode): {str(e)[:50]}"
        }

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
        # Cap at 1.0 (100%) to prevent exceeding maximum
        risk_score = min(risk_score, 1.0)
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

# --- Simulation Models ---

class SimulationRequest(BaseModel):
    tickers: List[str]
    start: Optional[str] = None  # Optional start date, default from config
    end: Optional[str] = None  # Optional end date, default latest
    shocked_node: str  # Must be in tickers universe
    shock_magnitude: float = Field(default=0.3, ge=0, le=1)  # 0..1
    connectivity_strength: float = Field(default=1.0, ge=0.1, le=2.0)
    liquidity_buffer_level: float = Field(default=1.0, ge=0.1, le=2.0)
    ccp_strictness: float = Field(default=0.5, ge=0, le=1)
    correlation_regime: float = Field(default=0.7, ge=0, le=1)
    steps: int = Field(default=10, ge=1, le=50)  # Cascade steps

class CongestionOutput(BaseModel):
    level: str  # "low", "medium", "high"
    most_congested_node: str
    max_score: float

class CascadeOutput(BaseModel):
    status: str  # "stable" or "cascade"
    failed_nodes: List[str]
    failed_edges: List[Dict[str, str]]  # [{"source": "A.NS", "target": "B.NS"}]
    failure_ratio: float
    cascade_depth: int
    analysis: str  # LLM explanation

class SimulationResponse(BaseModel):
    end_date: str
    used_tickers: List[str]
    masked_tickers: List[str]
    latest_features: Dict[str, float]
    latest_payoff_S: float
    predicted_next_systemic_risk: float
    congestion: CongestionOutput
    cascade: CascadeOutput
    ccp_funds: CCPFundsOutput

# --- Simulation Endpoint ---
@app.post("/simulate", response_model=SimulationResponse)
async def simulate(request: SimulationRequest):
    """
    Run a network stress simulation with node shocks and global controls.
    """
    if not model or not scaler:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    all_tickers = config["tickers"]
    valid_tickers = set(all_tickers)
    
    # Validate tickers
    tickers_subset = request.tickers
    if not tickers_subset:
        raise HTTPException(status_code=400, detail="Tickers list cannot be empty.")
    
    invalid = [t for t in tickers_subset if t not in valid_tickers]
    if invalid:
        raise HTTPException(status_code=400, detail=f"Invalid tickers: {invalid}. Must be in config.")
    
    # Validate shocked_node
    if request.shocked_node not in valid_tickers:
        raise HTTPException(status_code=400, detail=f"shocked_node '{request.shocked_node}' not in allowed universe.")
    
    # Fetch data
    start_date = request.start if request.start else config["start"]
    delta_ccp = config["delta_ccp"]
    ret_window = config.get("ret_window", 20)
    lookback = config.get("lookback", 20)
    
    try:
        end_param = request.end if request.end else None
        raw_data = yf.download(all_tickers, start=start_date, end=end_param, progress=False, auto_adjust=True, threads=False)
        if raw_data is None or raw_data.empty:
            raise HTTPException(status_code=500, detail="No data returned from yfinance.")
        data = raw_data['Close']
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch data: {e}")
    
    if isinstance(data, pd.Series):
        data = data.to_frame()
    
    existing_tickers = [t for t in all_tickers if t in data.columns]
    data = data[existing_tickers]
    N = len(existing_tickers)
    
    # Log returns
    returns = np.log(data / data.shift(1)).dropna()
    
    if len(returns) < ret_window:
        raise HTTPException(status_code=400, detail=f"Insufficient data. Need at least {ret_window} days.")
    
    # Use latest window for correlation
    latest_returns = returns.tail(ret_window)
    end_date = returns.index[-1].strftime("%Y-%m-%d")
    
    # --- Step 1: Build baseline adjacency A ---
    corr_mat = latest_returns.corr().values
    corr_mat = np.nan_to_num(corr_mat)
    A = np.maximum(0, corr_mat)
    np.fill_diagonal(A, 0)
    
    # --- Step 2: Add CCP edges ---
    # CCP strictness modifies edge weight
    ccp_edge = delta_ccp * (1 + request.ccp_strictness)
    
    # Extend matrix for CCP
    A_ext = np.zeros((N+1, N+1))
    A_ext[:N, :N] = A
    A_ext[:N, N] = ccp_edge  # Bank -> CCP
    A_ext[N, :N] = ccp_edge  # CCP -> Bank
    
    # --- Step 3: Apply global sliders ---
    # Connectivity strength
    A_ext = request.connectivity_strength * A_ext
    
    # Correlation regime (density gating)
    tau = 1 - request.correlation_regime
    A_ext[A_ext < tau] = 0
    
    # --- Step 4: Build node risk vector ---
    # Compute rolling risk metrics
    rolling_std = returns.rolling(window=ret_window).std()
    log_prices = returns.cumsum()
    rolling_max = log_prices.rolling(window=ret_window, min_periods=1).max()
    drawdown = (log_prices - rolling_max).abs()
    raw_risk = rolling_std + drawdown
    
    # Get latest risk
    latest_risk = raw_risk.iloc[-1].values
    
    # Normalize to [0,1]
    risk_min = np.nanmin(latest_risk)
    risk_max = np.nanmax(latest_risk)
    if risk_max > risk_min:
        r = (latest_risk - risk_min) / (risk_max - risk_min)
    else:
        r = np.zeros(N)
    
    r = np.nan_to_num(r, nan=0.0)
    
    # Create mask for selected tickers
    mask_vector = np.array([1.0 if t in tickers_subset else 0.0 for t in existing_tickers])
    
    # Apply masking
    r_masked = r * mask_vector
    
    # Apply shock
    if request.shocked_node in existing_tickers:
        shock_idx = existing_tickers.index(request.shocked_node)
        r_masked[shock_idx] = np.clip(r_masked[shock_idx] + request.shock_magnitude, 0, 1)
    
    # Extend with CCP (risk = 0)
    r_ext = np.append(r_masked, 0.0)
    mask_ext = np.append(mask_vector, 0.0)  # CCP excluded from payoff
    
    # --- Step 5: Eigen computations ---
    eigvals, eigvecs = np.linalg.eigh(A_ext)
    lambda_max = float(eigvals[-1])
    v = eigvecs[:, -1]
    if np.sum(v) < 0:
        v = -v
    v = v / (np.linalg.norm(v) + 1e-9)  # Normalize
    
    # --- Step 6: Payoff and congestion ---
    # S = r^T A (m ⊙ v)
    masked_v = mask_ext * v
    Av = np.dot(A_ext, masked_v)
    S = float(np.dot(r_ext, Av))
    
    # Congestion scores - ONLY for selected tickers
    # Buffer baseline: 1 - r, scaled by liquidity
    b = (1 - r_ext) * request.liquidity_buffer_level
    b = np.maximum(b, 0.1)  # Minimum buffer
    
    # Congestion score per node (only for selected nodes)
    congestion_scores = np.zeros(N+1)
    selected_indices = [i for i in range(N) if mask_vector[i] == 1]
    
    for i in range(N+1):
        if i < N and mask_vector[i] == 1:  # Only compute for selected nodes
            influence = np.dot(A_ext[:, i], r_ext)
            congestion_scores[i] = (v[i] * influence) / (b[i] + 1e-6)
        else:
            congestion_scores[i] = -np.inf  # Ignore non-selected nodes
    
    # Find most congested among SELECTED nodes only
    if len(selected_indices) > 0:
        selected_congestion = [congestion_scores[i] for i in selected_indices]
        max_congestion = float(np.max(selected_congestion))
        most_congested_idx = selected_indices[int(np.argmax(selected_congestion))]
        most_congested_node = existing_tickers[most_congested_idx]
    else:
        max_congestion = 0.0
        most_congested_node = "None"
    
    if max_congestion < 1:
        congestion_level = "low"
    elif max_congestion < 1.5:
        congestion_level = "medium"
    else:
        congestion_level = "high"
    
    # --- Step 7: ML feature row and prediction ---
    mean_risk = float(np.mean(r_masked))
    max_risk = float(np.max(r_masked))
    std_risk = float(np.std(r_masked))
    
    # Features in order
    feature_row = {
        "lambda_max": lambda_max,
        "mean_risk": mean_risk,
        "max_risk": max_risk,
        "std_risk": std_risk,
        "S_lag1": S,  # Approximate for simulation
        "S_lag5": S   # Approximate for simulation
    }
    
    # Create lookback sequence (repeat feature row)
    feature_array = np.array([[feature_row.get(col, 0) for col in feature_columns] for _ in range(lookback)])
    
    # Scale
    try:
        X_scaled = scaler.transform(feature_array)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scaling failed: {e}")
    
    X_reshaped = X_scaled.reshape(1, lookback, len(feature_columns))
    
    # Predict
    try:
        prediction = model.predict(X_reshaped, verbose=0)
        risk_score = float(prediction[0][0])
        # Cap at 1.0 (100%) to prevent exceeding maximum
        risk_score = min(risk_score, 1.0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")
    
    # --- Step 8: CCP Funds calculation ---
    ccp_funds = compute_ccp_funds(
        systemic=risk_score,
        lambda_max=lambda_max,
        std_risk=std_risk,
        connectivity_scale=request.connectivity_strength,
        liquidity_buffer_scale=request.liquidity_buffer_level
    )
    
    # --- Step 9: LLM-Based Cascade Analysis ---
    # Use LLM with hard boundaries to determine cascade
    used_tickers_list = [t for t in existing_tickers if t in tickers_subset]
    
    cascade_result = analyze_cascade_with_llm(
        nodes=used_tickers_list,
        shocked_node=request.shocked_node,
        shock_magnitude=request.shock_magnitude,
        connectivity_strength=request.connectivity_strength,
        liquidity_buffer=request.liquidity_buffer_level,
        systemic_risk=risk_score,
        congestion_level=congestion_level
    )
    
    return {
        "end_date": end_date,
        "used_tickers": used_tickers_list,
        "masked_tickers": [t for t in all_tickers if t not in tickers_subset],
        "latest_features": feature_row,
        "latest_payoff_S": S,
        "predicted_next_systemic_risk": risk_score,
        "congestion": {
            "level": congestion_level,
            "most_congested_node": most_congested_node,
            "max_score": round(max_congestion, 4)
        },
        "cascade": cascade_result,
        "ccp_funds": ccp_funds
    }

# --- News Generation Endpoints ---

class NewsGenerateRequest(BaseModel):
    tickers: List[str]
    count: int = Field(default=10, ge=1, le=20)

class NewsItem(BaseModel):
    id: str
    ticker: str
    company_name: str
    headline: str
    summary: str
    sentiment: str  # positive, negative, neutral
    confidence: float  # 0-1
    timestamp: str
    is_breaking: bool = False

class NewsGenerateResponse(BaseModel):
    news: List[NewsItem]

class NewsDefaultRequest(BaseModel):
    ticker: str
    company_name: str
    default_magnitude: float = Field(ge=0, le=1)

class NewsDefaultResponse(BaseModel):
    news: NewsItem

def generate_news_with_llm(tickers: List[str], count: int, is_default: bool = False, default_ticker: str = None, default_company: str = None, default_magnitude: float = 0.0) -> List[Dict[str, Any]]:
    """Generate realistic financial news using LLM with sentiment analysis."""
    
    # Get company names from tickers
    ticker_to_name = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            ticker_to_name[ticker] = info.get('shortName', info.get('longName', ticker.replace('.NS', '')))
        except:
            ticker_to_name[ticker] = ticker.replace('.NS', '')
    
    if is_default:
        # Generate breaking default news
        prompt = f"""Generate a realistic breaking financial news headline and 2-sentence summary about {default_company} ({default_ticker}) facing a severe default/liquidity crisis in the Indian market.

The news should be dramatic and reflect a severity level of {int(default_magnitude * 100)}%.

Respond in this exact JSON format:
{{
  "headline": "BREAKING: [Urgent headline about default/crisis]",
  "summary": "[2-sentence detailed summary explaining the situation]",
  "sentiment": "negative",
  "confidence": [0.85-0.99]
}}"""
    else:
        # Generate regular market news
        companies_str = ", ".join([f"{name} ({ticker})" for ticker, name in list(ticker_to_name.items())[:5]])
        prompt = f"""Generate {count} realistic financial news headlines for Indian companies: {companies_str}.

Create diverse news covering:
- Quarterly earnings (positive/negative)
- Market movements
- Business expansions
- Regulatory updates
- Industry trends

For EACH news item, classify sentiment as positive, negative, or neutral with a confidence score (0-1).

Respond in this exact JSON array format:
[
  {{
    "ticker": "TCS.NS",
    "headline": "[Realistic headline]",
    "summary": "[1-2 sentence summary]",
    "sentiment": "positive|negative|neutral",
    "confidence": 0.XX
  }}
]"""
    
    try:
        response = llm_client.chat.completions.create(
            model="Qwen/Qwen2.5-Coder-32B-Instruct",
            messages=[
                {"role": "system", "content": "You are a financial news generator for Indian markets. Generate realistic, professional news in JSON format only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=1500
        )
        
        content = response.choices[0].message.content.strip()
        
        # Extract JSON from response
        if '```json' in content:
            content = content.split('```json')[1].split('```')[0].strip()
        elif '```' in content:
            content = content.split('```')[1].split('```')[0].strip()
        
        news_data = json.loads(content)
        
        if is_default:
            # Single breaking news item
            return [{
                "ticker": default_ticker,
                "company_name": default_company,
                "headline": news_data["headline"],
                "summary": news_data["summary"],
                "sentiment": "negative",
                "confidence": news_data.get("confidence", 0.95),
                "is_breaking": True
            }]
        else:
            # Multiple news items
            news_items = []
            for item in news_data[:count]:
                ticker = item.get("ticker", tickers[0])
                news_items.append({
                    "ticker": ticker,
                    "company_name": ticker_to_name.get(ticker, ticker.replace('.NS', '')),
                    "headline": item["headline"],
                    "summary": item.get("summary", ""),
                    "sentiment": item["sentiment"],
                    "confidence": item.get("confidence", 0.75),
                    "is_breaking": False
                })
            return news_items
            
    except Exception as e:
        logger.error(f"News generation failed: {e}")
        # Fallback to template news
        if is_default:
            return [{
                "ticker": default_ticker,
                "company_name": default_company,
                "headline": f"BREAKING: {default_company} Faces Severe Liquidity Crisis",
                "summary": f"{default_company} has declared inability to meet short-term obligations amid market stress. Regulators have been notified.",
                "sentiment": "negative",
                "confidence": 0.90,
                "is_breaking": True
            }]
        else:
            return [
                {
                    "ticker": ticker,
                    "company_name": ticker_to_name.get(ticker, ticker.replace('.NS', '')),
                    "headline": f"{ticker_to_name.get(ticker, ticker.replace('.NS', ''))} Reports Stable Quarter",
                    "summary": "Company performance in line with market expectations.",
                    "sentiment": "neutral",
                    "confidence": 0.70,
                    "is_breaking": False
                }
                for ticker in tickers[:count]
            ]

@app.post("/news/generate", response_model=NewsGenerateResponse)
def generate_news(request: NewsGenerateRequest):
    """Generate realistic financial news feed for given tickers."""
    
    news_data = generate_news_with_llm(request.tickers, request.count)
    
    news_items = []
    for idx, item in enumerate(news_data):
        news_items.append(NewsItem(
            id=f"news-{datetime.now().timestamp()}-{idx}",
            ticker=item["ticker"],
            company_name=item["company_name"],
            headline=item["headline"],
            summary=item["summary"],
            sentiment=item["sentiment"],
            confidence=item["confidence"],
            timestamp=datetime.now().isoformat(),
            is_breaking=False
        ))
    
    return NewsGenerateResponse(news=news_items)

@app.post("/news/default", response_model=NewsDefaultResponse)
def generate_default_news(request: NewsDefaultRequest):
    """Generate breaking news for bank default event."""
    
    news_data = generate_news_with_llm(
        [request.ticker],
        1,
        is_default=True,
        default_ticker=request.ticker,
        default_company=request.company_name,
        default_magnitude=request.default_magnitude
    )
    
    item = news_data[0]
    
    return NewsDefaultResponse(
        news=NewsItem(
            id=f"breaking-{datetime.now().timestamp()}",
            ticker=item["ticker"],
            company_name=item["company_name"],
            headline=item["headline"],
            summary=item["summary"],
            sentiment="negative",
            confidence=item["confidence"],
            timestamp=datetime.now().isoformat(),
            is_breaking=True
        )
    )
