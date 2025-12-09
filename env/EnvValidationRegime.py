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

class StockEnvValidationRegime(gym.Env):
    """Validation environment with regime awareness"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, day=0, turbulence_threshold=140, iteration='', regime_detector=None):
        self.day = day
        self.df = df
        self.turbulence_threshold = turbulence_threshold
        self.iteration = iteration
        self.regime_detector = regime_detector or MarketRegimeDetector()
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(STOCK_DIM,))
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(185,))
        
        self.data = self.df.loc[self.day, :]
        self.terminal = False
        
        self.state = self._build_state()
        
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.rewards_memory = []
        self.current_regime = 'neutral'
        self.regime_memory = []
        
        self._seed()

    def _get_market_history(self, lookback=60):
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
        market_prices = self._get_market_history()
        self.current_regime = self.regime_detector.detect_regime(market_prices)
        regime_confidence = self.regime_detector.calculate_regime_confidence(market_prices)
        
        regime_bull = 1.0 if self.current_regime == 'bull' else 0.0
        regime_bear = 1.0 if self.current_regime == 'bear' else 0.0
        regime_neutral = 1.0 if self.current_regime == 'neutral' else 0.0
        
        base_state = [INITIAL_ACCOUNT_BALANCE] + \
                     self.data.adjcp.values.tolist() + \
                     [0] * STOCK_DIM + \
                     self.data.macd.values.tolist() + \
                     self.data.rsi.values.tolist() + \
                     self.data.cci.values.tolist() + \
                     self.data.adx.values.tolist()
        
        return base_state + [regime_bull, regime_bear, regime_neutral, regime_confidence]

    def _get_regime_adjusted_threshold(self):
        """Adjust turbulence threshold based on regime"""
        if self.current_regime == 'bull':
            return self.turbulence_threshold * 1.3
        elif self.current_regime == 'bear':
            return self.turbulence_threshold * 0.7
        else:
            return self.turbulence_threshold

    def _sell_stock(self, index, action):
        adjusted_threshold = self._get_regime_adjusted_threshold()
        
        if self.turbulence < adjusted_threshold:
            if self.state[index + STOCK_DIM + 1] > 0:
                self.state[0] += self.state[index + 1] * min(abs(action), self.state[index + STOCK_DIM + 1]) * (1 - TRANSACTION_FEE_PERCENT)
                self.state[index + STOCK_DIM + 1] -= min(abs(action), self.state[index + STOCK_DIM + 1])
                self.cost += self.state[index + 1] * min(abs(action), self.state[index + STOCK_DIM + 1]) * TRANSACTION_FEE_PERCENT
                self.trades += 1
        else:
            if self.state[index + STOCK_DIM + 1] > 0:
                self.state[0] += self.state[index + 1] * self.state[index + STOCK_DIM + 1] * (1 - TRANSACTION_FEE_PERCENT)
                self.state[index + STOCK_DIM + 1] = 0
                self.cost += self.state[index + 1] * self.state[index + STOCK_DIM + 1] * TRANSACTION_FEE_PERCENT
                self.trades += 1

    def _buy_stock(self, index, action):
        adjusted_threshold = self._get_regime_adjusted_threshold()
        
        if self.turbulence < adjusted_threshold:
            available_amount = self.state[0] // self.state[index + 1]
            self.state[0] -= self.state[index + 1] * min(available_amount, action) * (1 + TRANSACTION_FEE_PERCENT)
            self.state[index + STOCK_DIM + 1] += min(available_amount, action)
            self.cost += self.state[index + 1] * min(available_amount, action) * TRANSACTION_FEE_PERCENT
            self.trades += 1

    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1

        if self.terminal:
            plt.plot(self.asset_memory, 'r')
            plt.savefig(f'results/account_value_validation_regime_{self.iteration}.png')
            plt.close()
            
            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.to_csv(f'results/account_value_validation_{self.iteration}.csv')
            
            end_total_asset = self.state[0] + sum(np.array(self.state[1:(STOCK_DIM+1)]) * np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))
            
            df_total_value.columns = ['account_value']
            df_total_value['daily_return'] = df_total_value.pct_change(1)
            sharpe = (4**0.5) * df_total_value['daily_return'].mean() / df_total_value['daily_return'].std()
            
            return self.state, self.reward, self.terminal, {}

        actions = actions * HMAX_NORMALIZE
        adjusted_threshold = self._get_regime_adjusted_threshold()
        
        if self.turbulence >= adjusted_threshold:
            actions = np.array([-HMAX_NORMALIZE] * STOCK_DIM)
        
        begin_total_asset = self.state[0] + sum(np.array(self.state[1:(STOCK_DIM+1)]) * np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))
        
        argsort_actions = np.argsort(actions)
        sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
        buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

        for index in sell_index:
            self._sell_stock(index, actions[index])
        for index in buy_index:
            self._buy_stock(index, actions[index])

        self.day += 1
        self.data = self.df.loc[self.day, :]
        self.turbulence = self.data['turbulence'].values[0]
        self.state = self._build_state()
        self.regime_memory.append(self.current_regime)
        
        end_total_asset = self.state[0] + sum(np.array(self.state[1:(STOCK_DIM+1)]) * np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))
        self.asset_memory.append(end_total_asset)
        
        self.reward = (end_total_asset - begin_total_asset) * REWARD_SCALING
        self.rewards_memory.append(self.reward)

        return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False
        self.rewards_memory = []
        self.regime_memory = []
        self.state = self._build_state()
        return self.state

    def render(self, mode='human', close=False):
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
