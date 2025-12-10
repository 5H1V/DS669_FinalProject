import datetime
import os

# Use timestamp in a Windows-safe format
now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Define directories
TRAINED_MODEL_DIR = os.path.join("trained_models", now)

HMAX_NORMALIZE = 100
INITIAL_ACCOUNT_BALANCE = 1000000
STOCK_DIM = 30
TRANSACTION_FEE_PERCENT = 0.001
REWARD_SCALING = 1e-4

# Create folders if they don’t exist
os.makedirs(TRAINED_MODEL_DIR, exist_ok=True)