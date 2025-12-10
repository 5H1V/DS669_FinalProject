#!/usr/bin/env python3
"""
COMPLETE STANDALONE EFFICIENCY RATIO ADAPTIVE ENSEMBLE TRADING SYSTEM

This file contains EVERYTHING you need:
- Efficiency Ratio calculator
- All 3 gym environments (Train/Validation/Trade)
- All 3 training functions (A2C/PPO/DDPG)
- Complete ensemble strategy
- Visualization tools

ONLY REQUIREMENTS:
1. This file
2. done_data_2025.csv (your dataset)
3. pip install stable-baselines numpy pandas gym matplotlib scipy

USAGE:
    python complete_standalone.py
"""

import numpy as np
import pandas as pd
import time
import gym
from gym.utils import seeding
from gym import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from stable_baselines import A2C, PPO2, DDPG
from stable_baselines.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines.common.vec_env import DummyVecEnv

# Create results directory
os.makedirs('results', exist_ok=True)
os.makedirs('trained_models', exist_ok=True)

# ==============================================================================
# CONSTANTS
# ==============================================================================

HMAX_NORMALIZE = 100
INITIAL_ACCOUNT_BALANCE = 1000000
STOCK_DIM = 30
TRANSACTION_FEE_PERCENT = 0.001
REWARD_SCALING = 1e-4

# ==============================================================================
# PART 1: EFFICIENCY RATIO SYSTEM
# ==============================================================================

class EfficiencyRatioCalculator:
    def __init__(self, lookback=20):
        self.lookback = lookback
        
    def calculate_er(self, prices):
        if len(prices) < 2:
            return 0.5
        net_change = abs(prices[-1] - prices[0])
        total_change = np.sum(np.abs(np.diff(prices)))
        if total_change == 0:
            return 0.5
        return np.clip(net_change / total_change, 0.0, 1.0)
    
    def calculate_er_from_dataframe(self, df, date_col='datadate', price_col='adjcp'):
        """Calculate ER using market average prices"""
        unique_dates = sorted(df[date_col].unique())
        lookback = min(self.lookback, len(unique_dates))
        recent_dates = unique_dates[-lookback:]
        avg_prices = []
        
        for date in recent_dates:
            avg_price = df[df[date_col] == date][price_col].mean()
            avg_prices.append(avg_price)
        
        return self.calculate_er(np.array(avg_prices))
    
    def get_market_state(self, er):
        """Classify market: trending/mixed/choppy"""
        if er >= 0.7:
            return 'trending', 'low_complexity'
        elif er >= 0.4:
            return 'mixed', 'medium_complexity'
        else:
            return 'choppy', 'high_complexity'


class AdaptiveThresholdManager:    
    def __init__(self, base_turbulence_threshold=140.0, er_lookback=20):
        self.base_threshold = base_turbulence_threshold
        self.er_calculator = EfficiencyRatioCalculator(lookback=er_lookback)
        self.threshold_multipliers = {
            'low_complexity': 0.7,
            'medium_complexity': 1.0,
            'high_complexity': 1.3
        }
        self.er_history = []
        self.threshold_history = []
        self.market_state_history = []
    
    def calculate_adaptive_threshold(self, df, current_date=None):
        if current_date is not None:
            historical_df = df[df['datadate'] <= current_date]
        else:
            historical_df = df
        
        er = self.er_calculator.calculate_er_from_dataframe(historical_df)
        market_state, complexity = self.er_calculator.get_market_state(er)        
        multiplier = self.threshold_multipliers.get(complexity, 1.0)
        adaptive_threshold = self.base_threshold * multiplier
        
        self.er_history.append(er)
        self.threshold_history.append(adaptive_threshold)
        self.market_state_history.append(market_state)
        
        return adaptive_threshold, er, market_state, complexity


# ==============================================================================
# PART 2: GYM TRADING ENVIRONMENTS
# ==============================================================================

class StockEnvTrain(gym.Env):
    """Training environment"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, day=0):
        self.day = day
        self.df = df
        self.action_space = spaces.Box(low=-1, high=1, shape=(STOCK_DIM,))
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(181,))
        
        self.data = self.df.loc[self.day, :]
        self.terminal = False
        
        self.state = [INITIAL_ACCOUNT_BALANCE] + \
                     self.data.adjcp.values.tolist() + \
                     [0] * STOCK_DIM + \
                     self.data.macd.values.tolist() + \
                     self.data.rsi.values.tolist() + \
                     self.data.cci.values.tolist() + \
                     self.data.adx.values.tolist()
        
        self.reward = 0
        self.cost = 0
        self.trades = 0
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.rewards_memory = []
        self._seed()

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
            return self.state, self.reward, self.terminal, {}

        actions = actions * HMAX_NORMALIZE
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
        self.state = [self.state[0]] + \
                    self.data.adjcp.values.tolist() + \
                    list(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]) + \
                    self.data.macd.values.tolist() + \
                    self.data.rsi.values.tolist() + \
                    self.data.cci.values.tolist() + \
                    self.data.adx.values.tolist()
        
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
        self.state = [INITIAL_ACCOUNT_BALANCE] + \
                     self.data.adjcp.values.tolist() + \
                     [0] * STOCK_DIM + \
                     self.data.macd.values.tolist() + \
                     self.data.rsi.values.tolist() + \
                     self.data.cci.values.tolist() + \
                     self.data.adx.values.tolist()
        return self.state

    def render(self, mode='human'):
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


class StockEnvValidation(gym.Env):
    """Validation environment"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, day=0, turbulence_threshold=140, iteration=''):
        self.day = day
        self.df = df
        self.turbulence_threshold = turbulence_threshold
        self.iteration = iteration
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(STOCK_DIM,))
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(181,))
        
        self.data = self.df.loc[self.day, :]
        self.terminal = False
        
        self.state = [INITIAL_ACCOUNT_BALANCE] + \
                     self.data.adjcp.values.tolist() + \
                     [0] * STOCK_DIM + \
                     self.data.macd.values.tolist() + \
                     self.data.rsi.values.tolist() + \
                     self.data.cci.values.tolist() + \
                     self.data.adx.values.tolist()
        
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.rewards_memory = []
        self._seed()

    def _sell_stock(self, index, action):
        if self.turbulence < self.turbulence_threshold:
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
        if self.turbulence < self.turbulence_threshold:
            available_amount = self.state[0] // self.state[index + 1]
            self.state[0] -= self.state[index + 1] * min(available_amount, action) * (1 + TRANSACTION_FEE_PERCENT)
            self.state[index + STOCK_DIM + 1] += min(available_amount, action)
            self.cost += self.state[index + 1] * min(available_amount, action) * TRANSACTION_FEE_PERCENT
            self.trades += 1

    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1

        if self.terminal:
            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.to_csv(f'results/account_value_validation_{self.iteration}.csv')
            return self.state, self.reward, self.terminal, {}

        actions = actions * HMAX_NORMALIZE
        if self.turbulence >= self.turbulence_threshold:
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
        
        self.state = [self.state[0]] + \
                    self.data.adjcp.values.tolist() + \
                    list(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]) + \
                    self.data.macd.values.tolist() + \
                    self.data.rsi.values.tolist() + \
                    self.data.cci.values.tolist() + \
                    self.data.adx.values.tolist()
        
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
        self.state = [INITIAL_ACCOUNT_BALANCE] + \
                     self.data.adjcp.values.tolist() + \
                     [0] * STOCK_DIM + \
                     self.data.macd.values.tolist() + \
                     self.data.rsi.values.tolist() + \
                     self.data.cci.values.tolist() + \
                     self.data.adx.values.tolist()
        return self.state

    def render(self, mode='human', close=False):
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


class StockEnvTrade(gym.Env):
    """Trading environment with state continuity"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, day=0, turbulence_threshold=140, initial=True, 
                 previous_state=[], model_name='', iteration=''):
        self.day = day
        self.df = df
        self.initial = initial
        self.previous_state = previous_state
        self.turbulence_threshold = turbulence_threshold
        self.model_name = model_name
        self.iteration = iteration
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(STOCK_DIM,))
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(181,))
        
        self.data = self.df.loc[self.day, :]
        self.terminal = False
        
        self.state = [INITIAL_ACCOUNT_BALANCE] + \
                     self.data.adjcp.values.tolist() + \
                     [0] * STOCK_DIM + \
                     self.data.macd.values.tolist() + \
                     self.data.rsi.values.tolist() + \
                     self.data.cci.values.tolist() + \
                     self.data.adx.values.tolist()
        
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.rewards_memory = []
        self._seed()

    def _sell_stock(self, index, action):
        if self.turbulence < self.turbulence_threshold:
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
        if self.turbulence < self.turbulence_threshold:
            available_amount = self.state[0] // self.state[index + 1]
            self.state[0] -= self.state[index + 1] * min(available_amount, action) * (1 + TRANSACTION_FEE_PERCENT)
            self.state[index + STOCK_DIM + 1] += min(available_amount, action)
            self.cost += self.state[index + 1] * min(available_amount, action) * TRANSACTION_FEE_PERCENT
            self.trades += 1

    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1

        if self.terminal:
            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.to_csv(f'results/account_value_trade_{self.model_name}_{self.iteration}.csv')
            
            end_total_asset = self.state[0] + sum(np.array(self.state[1:(STOCK_DIM+1)]) * np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))
            print(f"End total asset: {end_total_asset:,.2f}")
            
            return self.state, self.reward, self.terminal, {}

        actions = actions * HMAX_NORMALIZE
        if self.turbulence >= self.turbulence_threshold:
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
        
        self.state = [self.state[0]] + \
                    self.data.adjcp.values.tolist() + \
                    list(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]) + \
                    self.data.macd.values.tolist() + \
                    self.data.rsi.values.tolist() + \
                    self.data.cci.values.tolist() + \
                    self.data.adx.values.tolist()
        
        end_total_asset = self.state[0] + sum(np.array(self.state[1:(STOCK_DIM+1)]) * np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))
        self.asset_memory.append(end_total_asset)
        self.reward = (end_total_asset - begin_total_asset) * REWARD_SCALING
        self.rewards_memory.append(self.reward)
        return self.state, self.reward, self.terminal, {}

    def reset(self):
        if self.initial:
            self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
            self.day = 0
            self.data = self.df.loc[self.day, :]
        else:
            previous_total_asset = self.previous_state[0] + sum(
                np.array(self.previous_state[1:(STOCK_DIM+1)]) * 
                np.array(self.previous_state[(STOCK_DIM+1):(STOCK_DIM*2+1)])
            )
            self.asset_memory = [previous_total_asset]
            self.day = 0
            self.data = self.df.loc[self.day, :]
        
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False
        self.rewards_memory = []
        
        if self.initial or not self.previous_state:
            self.state = [INITIAL_ACCOUNT_BALANCE] + \
                        self.data.adjcp.values.tolist() + \
                        [0] * STOCK_DIM + \
                        self.data.macd.values.tolist() + \
                        self.data.rsi.values.tolist() + \
                        self.data.cci.values.tolist() + \
                        self.data.adx.values.tolist()
        else:
            self.state = [self.previous_state[0]] + \
                        self.data.adjcp.values.tolist() + \
                        self.previous_state[(STOCK_DIM+1):(STOCK_DIM*2+1)] + \
                        self.data.macd.values.tolist() + \
                        self.data.rsi.values.tolist() + \
                        self.data.cci.values.tolist() + \
                        self.data.adx.values.tolist()
        
        return self.state

    def render(self, mode='human', close=False):
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


# ==============================================================================
# PART 3: TRAINING FUNCTIONS
# ==============================================================================

def train_A2C(env_train, model_name, timesteps=25000):
    """Train A2C model"""
    start = time.time()
    model = A2C('MlpPolicy', env_train, verbose=0)
    model.learn(total_timesteps=timesteps)
    end = time.time()
    model.save(f"trained_models/{model_name}")
    print(f'  Training time (A2C): {(end - start) / 60:.2f} minutes')
    return model


def train_PPO(env_train, model_name, timesteps=50000):
    """Train PPO model"""
    start = time.time()
    model = PPO2('MlpPolicy', env_train, ent_coef=0.005, nminibatches=8)
    model.learn(total_timesteps=timesteps)
    end = time.time()
    model.save(f"trained_models/{model_name}")
    print(f'  Training time (PPO): {(end - start) / 60:.2f} minutes')
    return model


def train_DDPG(env_train, model_name, timesteps=10000):
    """Train DDPG model"""
    n_actions = env_train.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions),
        sigma=float(0.5) * np.ones(n_actions)
    )
    
    start = time.time()
    model = DDPG('MlpPolicy', env_train, param_noise=None, action_noise=action_noise)
    model.learn(total_timesteps=timesteps)
    end = time.time()
    
    model.save(f"trained_models/{model_name}")
    print(f'  Training time (DDPG): {(end - start) / 60:.2f} minutes')
    return model


# ==============================================================================
# PART 4: HELPER FUNCTIONS
# ==============================================================================

def data_split(df, start, end):
    """Split data by date range"""
    data = df[(df.datadate >= start) & (df.datadate < end)]
    data = data.sort_values(['datadate', 'tic'], ignore_index=True)
    data.index = data.datadate.factorize()[0]
    return data


def DRL_validation(model, test_data, test_env, test_obs):
    """Run validation"""
    for i in range(len(test_data.index.unique())):
        action, _states = model.predict(test_obs)
        test_obs, rewards, dones, info = test_env.step(action)


def get_validation_sharpe(iteration):
    """Calculate Sharpe ratio"""
    try:
        df_total_value = pd.read_csv(f'results/account_value_validation_{iteration}.csv', index_col=0)
        df_total_value.columns = ['account_value_train']
        df_total_value['daily_return'] = df_total_value.pct_change(1)
        valid_returns = df_total_value['daily_return'].dropna()
        
        if len(valid_returns) == 0 or valid_returns.std() == 0:
            return 0.0
        
        sharpe = (4 ** 0.5) * valid_returns.mean() / valid_returns.std()
        return sharpe if not np.isnan(sharpe) else 0.0
    except:
        return 0.0


def DRL_prediction(df, model, name, last_state, iter_num, unique_trade_date,
                   rebalance_window, turbulence_threshold, initial):
    """Execute trading"""
    trade_data = data_split(df, 
                           start=unique_trade_date[iter_num - rebalance_window],
                           end=unique_trade_date[iter_num])
    
    env_trade = DummyVecEnv([lambda: StockEnvTrade(
        trade_data,
        turbulence_threshold=turbulence_threshold,
        initial=initial,
        previous_state=last_state,
        model_name=name,
        iteration=iter_num
    )])
    
    obs_trade = env_trade.reset()
    
    for i in range(len(trade_data.index.unique())):
        action, _states = model.predict(obs_trade)
        obs_trade, rewards, dones, info = env_trade.step(action)
        
        if i == (len(trade_data.index.unique()) - 2):
            last_state = env_trade.render()
    
    return last_state


# ==============================================================================
# PART 5: MAIN ENSEMBLE STRATEGY WITH ER ADAPTATION
# ==============================================================================

def run_er_adaptive_ensemble(data_file='done_data_2025.csv'):
    """
    MAIN FUNCTION: Run complete ER-adaptive ensemble strategy
    """
    
    print("="*80)
    print("EFFICIENCY RATIO ADAPTIVE ENSEMBLE TRADING SYSTEM")
    print("="*80)
    
    # Load data
    print("\n📂 Loading data...")
    data = pd.read_csv(data_file, index_col=0)
    print(f"✅ Loaded {len(data)} rows")
    print(f"   Date range: {data['datadate'].min()} to {data['datadate'].max()}")
    print(f"   Unique dates: {data['datadate'].nunique()}")
    print(f"   Stocks: {data['tic'].nunique()}")
    
    # Define trading period
    unique_trade_date = data[(data.datadate >= 20190207) & 
                            (data.datadate <= 20220616)].datadate.unique()
    print(f"\n📅 Trading period: {len(unique_trade_date)} days")
    
    # Parameters
    rebalance_window = 63
    validation_window = 63
    base_turbulence = 140.0
    
    # Check data sufficiency
    min_required = rebalance_window + validation_window
    if len(unique_trade_date) < min_required:
        print(f"\n⚠️  WARNING: Only {len(unique_trade_date)} days available")
        print(f"   Need at least {min_required} days")
        print("   Adjusting parameters...")
        rebalance_window = max(10, len(unique_trade_date) // 3)
        validation_window = max(10, len(unique_trade_date) // 3)
    
    print(f"\n⚙️  Parameters:")
    print(f"   Rebalance window: {rebalance_window} days")
    print(f"   Validation window: {validation_window} days")
    print(f"   Base turbulence threshold: {base_turbulence}")
    
    # Initialize ER manager
    threshold_manager = AdaptiveThresholdManager(
        base_turbulence_threshold=base_turbulence,
        er_lookback=20
    )
    
    # Tracking variables
    last_state_ensemble = []
    results = {
        'iteration': [],
        'model_selected': [],
        'ppo_sharpe': [],
        'a2c_sharpe': [],
        'ddpg_sharpe': [],
        'efficiency_ratio': [],
        'adaptive_threshold': [],
        'market_state': []
    }
    
    # Main loop
    start_time = time.time()
    iteration_count = 0
    
    for i in range(rebalance_window + validation_window, len(unique_trade_date), rebalance_window):
        iteration_count += 1
        
        print(f"\n{'='*80}")
        print(f"ITERATION {iteration_count} - Date: {unique_trade_date[i]}")
        print(f"{'='*80}")
        
        try:
            initial = (i - rebalance_window - validation_window == 0)
            current_date = unique_trade_date[i - rebalance_window - validation_window]
            
            # Calculate ER-adaptive threshold
            turbulence_threshold, er, market_state, complexity = \
                threshold_manager.calculate_adaptive_threshold(data, current_date)
            
            print(f"\n📊 MARKET ANALYSIS:")
            print(f"   ER: {er:.4f} | State: {market_state.upper()} | Threshold: {turbulence_threshold:.2f}")
            
            # Data splits
            train_data = data_split(data, start=20090000, 
                                   end=unique_trade_date[i - rebalance_window - validation_window])
            validation_data = data_split(data, 
                                        start=unique_trade_date[i - rebalance_window - validation_window],
                                        end=unique_trade_date[i - rebalance_window])
            
            if len(train_data) == 0 or len(validation_data) == 0:
                print("  ⚠️  Insufficient data, skipping...")
                continue
            
            # Environments
            env_train = DummyVecEnv([lambda: StockEnvTrain(train_data)])
            env_val = DummyVecEnv([lambda: StockEnvValidation(
                validation_data, turbulence_threshold=turbulence_threshold, iteration=i
            )])
            
            # Training
            print(f"\n🎓 TRAINING:")
            model_a2c = train_A2C(env_train, f"A2C_ER_{i}", timesteps=30000)
            model_ppo = train_PPO(env_train, f"PPO_ER_{i}", timesteps=100000)
            model_ddpg = train_DDPG(env_train, f"DDPG_ER_{i}", timesteps=10000)
            
            # Validation
            print(f"\n✅ VALIDATION:")
            obs_val = env_val.reset()
            DRL_validation(model_a2c, validation_data, env_val, obs_val)
            sharpe_a2c = get_validation_sharpe(i)
            
            obs_val = env_val.reset()
            DRL_validation(model_ppo, validation_data, env_val, obs_val)
            sharpe_ppo = get_validation_sharpe(i)
            
            obs_val = env_val.reset()
            DRL_validation(model_ddpg, validation_data, env_val, obs_val)
            sharpe_ddpg = get_validation_sharpe(i)
            
            print(f"   A2C: {sharpe_a2c:.4f} | PPO: {sharpe_ppo:.4f} | DDPG: {sharpe_ddpg:.4f}")
            
            # ER-weighted selection
            if er > 0.7:  # Trending
                weights = {'PPO': 1.0, 'A2C': 0.9, 'DDPG': 1.2}
            elif er < 0.4:  # Choppy
                weights = {'PPO': 1.2, 'A2C': 1.0, 'DDPG': 0.8}
            else:  # Mixed
                weights = {'PPO': 1.0, 'A2C': 1.0, 'DDPG': 1.0}
            
            weighted_sharpes = {
                'PPO': sharpe_ppo * weights['PPO'],
                'A2C': sharpe_a2c * weights['A2C'],
                'DDPG': sharpe_ddpg * weights['DDPG']
            }
            
            best_model_name = max(weighted_sharpes, key=weighted_sharpes.get)
            model_ensemble = {'PPO': model_ppo, 'A2C': model_a2c, 'DDPG': model_ddpg}[best_model_name]
            
            print(f"\n🏆 SELECTED: {best_model_name}")
            
            # Trading
            print(f"\n💰 TRADING...")
            last_state_ensemble = DRL_prediction(
                data, model_ensemble, best_model_name, last_state_ensemble,
                i, unique_trade_date, rebalance_window, turbulence_threshold, initial
            )
            
            # Store results
            results['iteration'].append(iteration_count)
            results['model_selected'].append(best_model_name)
            results['ppo_sharpe'].append(sharpe_ppo)
            results['a2c_sharpe'].append(sharpe_a2c)
            results['ddpg_sharpe'].append(sharpe_ddpg)
            results['efficiency_ratio'].append(er)
            results['adaptive_threshold'].append(turbulence_threshold)
            results['market_state'].append(market_state)
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final results
    end_time = time.time()
    
    print(f"\n{'='*80}")
    print(f"COMPLETE - {(end_time - start_time) / 60:.2f} minutes")
    print(f"Iterations: {iteration_count}")
    
    if iteration_count > 0:
        print(f"\n📊 AVERAGE SHARPE RATIOS:")
        print(f"   PPO: {np.mean(results['ppo_sharpe']):.4f}")
        print(f"   A2C: {np.mean(results['a2c_sharpe']):.4f}")
        print(f"   DDPG: {np.mean(results['ddpg_sharpe']):.4f}")
        
        print(f"\n🎯 ER STATISTICS:")
        print(f"   Mean: {np.mean(results['efficiency_ratio']):.4f}")
        print(f"   Range: {np.min(results['efficiency_ratio']):.4f} - {np.max(results['efficiency_ratio']):.4f}")
        
        # Save results
        df_results = pd.DataFrame(results)
        df_results.to_csv('results/ensemble_summary_er.csv', index=False)
        print(f"\n✅ Results saved to results/ensemble_summary_er.csv")
    
    print(f"{'='*80}\n")
    
    return results


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    
    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                                                                          ║
    ║   EFFICIENCY RATIO ADAPTIVE ENSEMBLE TRADING SYSTEM                      ║
    ║   Complete Standalone Version                                            ║
    ║                                                                          ║
    ║   Features:                                                              ║
    ║   ✓ Automatic market complexity detection (ER)                           ║
    ║   ✓ Dynamic threshold adaptation                                         ║
    ║   ✓ Three RL models (A2C, PPO, DDPG)                                     ║
    ║   ✓ Quarterly rebalancing                                                ║
    ║   ✓ Portfolio state continuity                                           ║
    ║   ✓ Zero parameter tuning required                                       ║
    ║                                                                          ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Check if dataset exists
    if not os.path.exists('done_data_2025.csv'):
        print("❌ ERROR: done_data_2025.csv not found!")
        print("   Please place your dataset in the same directory as this script.")
        exit(1)
    
    # Run strategy
    try:
        results = run_er_adaptive_ensemble('done_data_2025.csv')
        print("\n✅ Strategy execution complete!")
        print("   Check the 'results/' folder for outputs.")
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()