import numpy as np
import pandas as pd
import time

from stable_baselines import A2C, PPO2, DDPG
from stable_baselines.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines.common.vec_env import DummyVecEnv

from config.config import *
from env.EnvTradeRegime import *
from env.EnvTrainRegime import *
from env.EnvValidationRegime import *
from AdaptiveThreshold import *


def train_A2C(env_train, model_name, timesteps=25000):
    """Train A2C model"""
    start = time.time()
    model = A2C('MlpPolicy', env_train, verbose=0)
    model.learn(total_timesteps=timesteps)
    end = time.time()
    model.save(f"trained_models/{model_name}")
    print(f'Training time (A2C): {(end - start) / 60:.2f} minutes')
    return model


def train_PPO(env_train, model_name, timesteps=50000):
    """Train PPO model"""
    start = time.time()
    model = PPO2('MlpPolicy', env_train, ent_coef=0.005, nminibatches=8)
    model.learn(total_timesteps=timesteps)
    end = time.time()
    model.save(f"trained_models/{model_name}")
    print(f'Training time (PPO): {(end - start) / 60:.2f} minutes')
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
    
    model.save(f"{TRAINED_MODEL_DIR}/{model_name}")
    print(f'Training time (DDPG): {(end - start) / 60:.2f} minutes')
    return model

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

def run_er_adaptive_ensemble(data):

    print("\nLoading data...")
    print(f"Loaded {len(data)} rows")
    print(f"Date range: {data['datadate'].min()} to {data['datadate'].max()}")
    print(f"Unique dates: {data['datadate'].nunique()}")
    print(f"Stocks: {data['tic'].nunique()}")
    
    # Define trading period
    unique_trade_date = data[(data.datadate >= 20190207) & 
                            (data.datadate <= 20220616)].datadate.unique()
    print(f"\nTrading period: {len(unique_trade_date)} days")
    
    # Parameters
    rebalance_window = 63
    validation_window = 63
    base_turbulence = 140.0
    
    # Check data sufficiency
    min_required = rebalance_window + validation_window
    if len(unique_trade_date) < min_required:
        print(f"\nWARNING: Only {len(unique_trade_date)} days available")
        print(f"Need at least {min_required} days")
        print("Adjusting parameters...")
        rebalance_window = max(10, len(unique_trade_date) // 3)
        validation_window = max(10, len(unique_trade_date) // 3)
    
    print(f"\nParameters:")
    print(f"Rebalance window: {rebalance_window} days")
    print(f"Validation window: {validation_window} days")
    print(f"Base turbulence threshold: {base_turbulence}")
    
    # Initialize ER manager
    threshold_manager = AdaptiveThresholdManager(base_turbulence_threshold=base_turbulence, er_lookback=20)
    
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
        print('-'*40)
        print(f"Iteration: {iteration_count} | Date: {unique_trade_date[i]}")
        
        try:
            initial = (i - rebalance_window - validation_window == 0)
            current_date = unique_trade_date[i - rebalance_window - validation_window]
            
            turbulence_threshold, er, market_state, complexity = threshold_manager.calculate_adaptive_threshold(data, current_date)
            
            print(f"Results:")
            print(f"ER: {er:.4f} | State: {market_state.upper()} | Threshold: {turbulence_threshold:.2f}")
            
            # Data splits
            train_data = data_split(data, start=20090000, end=unique_trade_date[i - rebalance_window - validation_window])
            validation_data = data_split(data, start=unique_trade_date[i - rebalance_window - validation_window],
                                         end=unique_trade_date[i - rebalance_window])
            
            if len(train_data) == 0 or len(validation_data) == 0:
                continue
            
            # Environments
            env_train = DummyVecEnv([lambda: StockEnvTrain(train_data)])
            env_val = DummyVecEnv([lambda: StockEnvValidation(
                validation_data, turbulence_threshold=turbulence_threshold, iteration=i
            )])
            
            # Training
            print(f"\n TRAINING:")
            model_a2c = train_A2C(env_train, f"A2C_ER_{i}", timesteps=30000)
            model_ppo = train_PPO(env_train, f"PPO_ER_{i}", timesteps=100000)
            model_ddpg = train_DDPG(env_train, f"DDPG_ER_{i}", timesteps=10000)
            
            # Validation
            print(f"\n VALIDATION:")
            obs_val = env_val.reset()
            DRL_validation(model_a2c, validation_data, env_val, obs_val)
            sharpe_a2c = get_validation_sharpe(i)
            
            obs_val = env_val.reset()
            DRL_validation(model_ppo, validation_data, env_val, obs_val)
            sharpe_ppo = get_validation_sharpe(i)
            
            obs_val = env_val.reset()
            DRL_validation(model_ddpg, validation_data, env_val, obs_val)
            sharpe_ddpg = get_validation_sharpe(i)
            
            print(f"A2C: {sharpe_a2c:.4f} | PPO: {sharpe_ppo:.4f} | DDPG: {sharpe_ddpg:.4f}")
            
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
            
            print(f"\nSELECTED: {best_model_name}")
            
            # Trading
            print(f"\nTRADING...")
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
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final results
    end_time = time.time()
    
    print(f"\n{'='*80}")
    print(f"COMPLETE - {(end_time - start_time) / 60:.2f} minutes")
    print(f"Iterations: {iteration_count}")
    
    if iteration_count > 0:
        print(f"\nAVERAGE SHARPE RATIOS:")
        print(f"PPO: {np.mean(results['ppo_sharpe']):.4f}")
        print(f"A2C: {np.mean(results['a2c_sharpe']):.4f}")
        print(f"DDPG: {np.mean(results['ddpg_sharpe']):.4f}")
        
        print(f"\nER STATISTICS:")
        print(f"Mean: {np.mean(results['efficiency_ratio']):.4f}")
        print(f"Range: {np.min(results['efficiency_ratio']):.4f} - {np.max(results['efficiency_ratio']):.4f}")
        
        # Save results
        df_results = pd.DataFrame(results)
        df_results.to_csv('results/ensemble_summary_er.csv', index=False)
        print(f"\nResults saved to results/ensemble_summary_er.csv")
    
    print(f"{'-'*40}\n")
    
    return results

