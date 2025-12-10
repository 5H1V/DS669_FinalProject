import numpy as np

from config.config import *

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

