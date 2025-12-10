import os
import pandas as pd

from config.config import *
from model.regime_models import *

# Create results directory
os.makedirs('results', exist_ok=True)
os.makedirs('trained_models', exist_ok=True)

if __name__ == "__main__":
    data = pd.read_csv('done_data_2025.csv', index_col=0)
    results = run_er_adaptive_ensemble(data)
    print(results)
