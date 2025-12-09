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

from MarketRegimeDetector import MarketRegimeDetector

# Constants
HMAX_NORMALIZE = 100
INITIAL_ACCOUNT_BALANCE = 1000000
STOCK_DIM = 30
TRANSACTION_FEE_PERCENT = 0.001
REWARD_SCALING = 1e-4

class StockEnvTrainRegime(gym.Env):
    """Training environment with regime awareness"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, day=0, regime_detector=None):
        self.day = day
        self.df = df
        self.regime_detector = regime_detector or MarketRegimeDetector()
        
        # Action and observation spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(STOCK_DIM,))
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(185,))
        
        # Load data
        self.data = self.df.loc[self.day, :]
        self.terminal = False
        
        # Initialize state
        self.state = self._build_state()
        
        # Tracking
        self.reward = 0
        self.cost = 0
        self.trades = 0
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.rewards_memory = []
        self.current_regime = 'neutral'
        
        self._seed()

    def _get_market_history(self, lookback=60):
        """Get historical market prices for regime detection"""
        if self.day < lookback:
            start_day = 0
        else:
            start_day = self.day - lookback
        
        history = []
        for d in range(start_day, self.day + 1):
            day_data = self.df.loc[d, :]
            avg_price = np.mean(day_data.adjcp.values)
            history.append(avg_price)
        
        return np.array(history)

    def _build_state(self):
        """Build state vector with regime information"""
        market_prices = self._get_market_history()
        self.current_regime = self.regime_detector.detect_regime(market_prices)
        regime_confidence = self.regime_detector.calculate_regime_confidence(market_prices)
        
        # One-hot encode regime
        regime_bull = 1.0 if self.current_regime == 'bull' else 0.0
        regime_bear = 1.0 if self.current_regime == 'bear' else 0.0
        regime_neutral = 1.0 if self.current_regime == 'neutral' else 0.0
        
        # Base state: 181 dimensions
        base_state = [INITIAL_ACCOUNT_BALANCE] + \
                     self.data.adjcp.values.tolist() + \
                     [0] * STOCK_DIM + \
                     self.data.macd.values.tolist() + \
                     self.data.rsi.values.tolist() + \
                     self.data.cci.values.tolist() + \
                     self.data.adx.values.tolist()
        
        # Add regime features: 4 dimensions
        return base_state + [regime_bull, regime_bear, regime_neutral, regime_confidence]

    def _sell_stock(self, index, action):
        if self.state[index + STOCK_DIM + 1] > 0:
            self.state[0] += self.state[index + 1] * min(abs(action), self.state[index + STOCK_DIM + 1]) * (1 - TRANSACTION_FEE_PERCENT)
            self.state[index + STOCK_DIM + 1] -= min(abs(action), self.state[index + STOCK_DIM + 1])
            self.cost += self.state[index + 1] * min(abs(action), self.state[index + STOCK_DIM + 1]) * TRANSACTION_FEE_PERCENT
            self.trades += 1

    def _buy_stock(self, index, action):
        available_amount = self.state[0] // self.state[index + 1]
        self.state[0] -= self.state[index + 1] * min(available_amount, action) * (1 + TRANSACTION_FEE_PERCENT)
        self.state[index + STOCK_DIM + 1] += min(available_amount, action)
        self.cost += self.state[index + 1] * min(available_amount, action) * TRANSACTION_FEE_PERCENT
        self.trades += 1

    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1

        if self.terminal:
            plt.plot(self.asset_memory, 'r')
            plt.savefig('results/account_value_train_regime.png')
            plt.close()
            
            end_total_asset = self.state[0] + sum(np.array(self.state[1:(STOCK_DIM+1)]) * np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))
            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.to_csv('results/account_value_train_regime.csv')
            
            return self.state, self.reward, self.terminal, {}

        # Scale actions
        actions = actions * HMAX_NORMALIZE
        
        begin_total_asset = self.state[0] + sum(np.array(self.state[1:(STOCK_DIM+1)]) * np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))
        
        # Execute trades
        argsort_actions = np.argsort(actions)
        sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
        buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

        for index in sell_index:
            self._sell_stock(index, actions[index])
        for index in buy_index:
            self._buy_stock(index, actions[index])

        # Move to next day
        self.day += 1
        self.data = self.df.loc[self.day, :]
        self.state = self._build_state()
        
        end_total_asset = self.state[0] + sum(np.array(self.state[1:(STOCK_DIM+1)]) * np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))
        self.asset_memory.append(end_total_asset)
        
        self.reward = (end_total_asset - begin_total_asset) * REWARD_SCALING
        self.rewards_memory.append(self.reward)

        return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.cost = 0
        self.trades = 0
        self.terminal = False
        self.rewards_memory = []
        self.state = self._build_state()
        return self.state

    def render(self, mode='human'):
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
