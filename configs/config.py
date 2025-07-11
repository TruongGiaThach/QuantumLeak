 
import os
import torch

# Paths
SAVE_PATH = "./saved_models"
QUANTUM_LEAK_SAVE_PATH = "./results"
DATA_PATH = "./data"

# Hyperparameters
N_EPOCHS = 30
N_LAYERS = 2
N_TRAIN = 25000 #5000
N_TEST = 5000   #1000
N_QUBITS = 4
BATCH_SIZE = 8
LEARNING_RATE = 0.001
PREPROCESS = True
N_LEAK_EPOCHS = 30 #30
N_LEAK_LAYERS = 2
N_LEAK_TRAIN = 50000
N_LEAK_TEST = 10000
N_LEAK_QUBITS = 4
LEAK_BATCH_SIZE = 8
LEAK_QUERY_BUDGET = 6000
LEAK_N_COMMITTEE = 3

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
QUANTUM_DEVICE = "lightning.gpu" if torch.cuda.is_available() else "default.qubit"

# Ensure directories exist
os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(QUANTUM_LEAK_SAVE_PATH, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(SAVE_PATH + "/quanv", exist_ok=True)
os.makedirs(SAVE_PATH + "/basic_qnn", exist_ok=True)
os.makedirs(SAVE_PATH + "/circuit14", exist_ok=True)
os.makedirs(SAVE_PATH + "/transfer_learning", exist_ok=True)