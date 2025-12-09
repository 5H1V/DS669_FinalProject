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

from run_Regime import data_split
from env.EnvTradeRegime import StockEnvTradeRegime

# Constants
HMAX_NORMALIZE = 100
INITIAL_ACCOUNT_BALANCE = 1000000
STOCK_DIM = 30
TRANSACTION_FEE_PERCENT = 0.001
REWARD_SCALING = 1e-4

def train_A2C(env_train, model_name, timesteps=25000):
    """A2C model with regime-aware state"""
    start = time.time()
    model = A2C('MlpPolicy', env_train, verbose=0)
    model.learn(total_timesteps=timesteps)
    end = time.time()
    model.save(f"trained_models/{model_name}")
    print(f'Training time (A2C): {(end - start) / 60:.2f} minutes')
    return model


def train_PPO(env_train, model_name, timesteps=50000):
    """PPO model with regime-aware state"""
    start = time.time()
    model = PPO2('MlpPolicy', env_train, ent_coef=0.005, nminibatches=8)
    model.learn(total_timesteps=timesteps)
    end = time.time()
    model.save(f"trained_models/{model_name}")
    print(f'Training time (PPO): {(end - start) / 60:.2f} minutes')
    return model


def train_DDPG(env_train, model_name, timesteps=10000):
    """DDPG model with regime-aware state"""
    n_actions = env_train.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions), 
        sigma=float(0.5) * np.ones(n_actions)
    )
    
    start = time.time()
    model = DDPG('MlpPolicy', env_train, param_noise=param_noise, action_noise=action_noise)
    model.learn(total_timesteps=timesteps)
    end = time.time()
    
    model.save(f"trained_models/{model_name}")
    print(f'Training time (DDPG): {(end - start) / 60:.2f} minutes')
    return model


# ==============================================================================
# PART 6: VALIDATION AND PREDICTION FUNCTIONS
# ==============================================================================

def DRL_validation(model, test_data, test_env, test_obs):
    """Validation process - same as original"""
    for i in range(len(test_data.index.unique())):
        action, _states = model.predict(test_obs)
        test_obs, rewards, dones, info = test_env.step(action)


def get_validation_sharpe(iteration):
    """Calculate Sharpe ratio based on validation results - same as original"""
    df_total_value = pd.read_csv(f'results/account_value_validation_{iteration}.csv', index_col=0)
    df_total_value.columns = ['account_value_train']
    df_total_value['daily_return'] = df_total_value.pct_change(1)
    sharpe = (4 ** 0.5) * df_total_value['daily_return'].mean() / \
             df_total_value['daily_return'].std()
    return sharpe


def DRL_prediction(df, model, name, last_state, iter_num, unique_trade_date,
                   rebalance_window, turbulence_threshold, initial, regime_detector):
    """Make predictions using trained model - UPDATED FOR REGIME"""
    
    # Prepare trading data
    trade_data = data_split(df, 
                           start=unique_trade_date[iter_num - rebalance_window],
                           end=unique_trade_date[iter_num])
    
    # Create trading environment with regime detection
    env_trade = DummyVecEnv([lambda: StockEnvTradeRegime(
        trade_data,
        turbulence_threshold=turbulence_threshold,
        initial=initial,
        previous_state=last_state,
        model_name=name,
        iteration=iter_num,
        regime_detector=regime_detector
    )])
    
    obs_trade = env_trade.reset()
    
    # Run through trading period
    for i in range(len(trade_data.index.unique())):
        action, _states = model.predict(obs_trade)
        obs_trade, rewards, dones, info = env_trade.step(action)
        
        if i == (len(trade_data.index.unique()) - 2):
            last_state = env_trade.render()
    
    # Save last state for continuity
    df_last_state = pd.DataFrame({'last_state': last_state})
    df_last_state.to_csv(f'results/last_state_{name}_{iter_num}.csv', index=False)
    
    return last_state
