import numpy as np
import pandas as pd
import gym
from gym.utils import seeding
from gym import spaces

from config.config import *

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
