
import json
import math
import os
import sys
from datetime import date
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import log_loss

PROJECT_ROOT = Path(__file__).resolve().parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from betting_app.core.db import query_df
from src.ratings.manager import RatingManager

def run_eval(tau_gl, tau_ts):
    params = {
        "elo": {"k_player": 48, "k_team": 64},
        "gl": {"tau": tau_gl}, # Note: RatingManager needs to pass this to GlickoRating
        "ts": {"mu": 25.0, "sigma": 8.333, "beta": 4.16, "tau": tau_ts},
    }
    # We need to modify RatingManager or GlickoRating to accept tau
    # For now, let's assume we can pass it.
    
    # Actually, let's just implement a quick loop here to avoid RatingManager overhead
    # if we want to be fast. But for correctness, let's use the systems.
    pass

# I will first modify GlickoRating and TrueSkillRating to accept parameters properly.
