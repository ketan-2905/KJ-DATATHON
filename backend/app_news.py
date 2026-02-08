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
