import numpy as np
import pandas as pd
import time
import gym
from gym.utils import seeding
from gym import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

from stable_baselines import A2C, PPO2, DDPG
from stable_baselines.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines.common.vec_env import DummyVecEnv

# Constants
HMAX_NORMALIZE = 100
INITIAL_ACCOUNT_BALANCE = 1000000
STOCK_DIM = 30
TRANSACTION_FEE_PERCENT = 0.001
REWARD_SCALING = 1e-4

class MarketRegimeDetector:
    """
    Detects market regimes using multiple methods:
    1. Price momentum (returns-based)
    2. Volatility clustering
    3. Trend strength analysis
    """
    
    def __init__(self, lookback_short=20, lookback_long=60):
        self.lookback_short = lookback_short
        self.lookback_long = lookback_long
        
    def detect_regime(self, price_history, volume_history=None):
        """
        Returns: 'bull', 'bear', or 'neutral'
        """
        if len(price_history) < self.lookback_long:
            return 'neutral'
        
        # Calculate returns
        returns = np.diff(price_history) / price_history[:-1]
        
        # 1. Momentum-based regime
        short_ma = np.mean(price_history[-self.lookback_short:])
        long_ma = np.mean(price_history[-self.lookback_long:])
        momentum_signal = (short_ma - long_ma) / long_ma
        
        # 2. Volatility-based regime
        recent_vol = np.std(returns[-self.lookback_short:])
        historical_vol = np.std(returns[-self.lookback_long:])
        
        # 3. Trend strength
        trend_strength = np.mean(returns[-self.lookback_short:])
        
        # 4. Risk-adjusted returns
        if recent_vol > 0:
            sharpe_recent = trend_strength / recent_vol
        else:
            sharpe_recent = 0
            
        # Regime classification
        if momentum_signal > 0.02 and trend_strength > 0 and sharpe_recent > 0.5:
            return 'bull'
        elif momentum_signal < -0.02 and trend_strength < 0 and sharpe_recent < -0.5:
            return 'bear'
        else:
            return 'neutral'
    
    def calculate_regime_confidence(self, price_history):
        """
        Returns confidence score (0-1) for current regime
        """
        if len(price_history) < self.lookback_long:
            return 0.5
        
        returns = np.diff(price_history) / price_history[:-1]
        
        # Measure consistency of returns
        recent_returns = returns[-self.lookback_short:]
        positive_days = np.sum(recent_returns > 0) / len(recent_returns)
        
        # Confidence is higher when returns are consistently directional
        confidence = abs(2 * positive_days - 1)
        
        return confidence
