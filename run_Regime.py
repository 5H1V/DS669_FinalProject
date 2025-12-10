import os

from config.config import *
from model.regime_models import *

# Create results directory
os.makedirs('results', exist_ok=True)
os.makedirs('trained_models', exist_ok=True)

if __name__ == "__main__":
    results = run_er_adaptive_ensemble('done_data_2025.csv')
    print(results)
