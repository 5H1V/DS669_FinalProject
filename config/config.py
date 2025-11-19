import datetime
import os

# Use timestamp in a Windows-safe format
now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Define directories
TRAINED_MODEL_DIR = os.path.join("trained_models", now)
TURBULENCE_DATA = "data/dow30_turbulence_index.csv"
TESTING_DATA_FILE = "test.csv"

# Create folders if they don’t exist
os.makedirs(TRAINED_MODEL_DIR, exist_ok=True)