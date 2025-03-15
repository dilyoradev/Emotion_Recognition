import torch

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# Dataset paths
TRAIN_DIR = "/Users/imaan/development/emotion_recognition/data/train"
TEST_DIR = "/Users/imaan/development/emotion_recognition/data/test"

# Training settings
BATCH_SIZE = 32
NUM_EPOCHS = 3
LEARNING_RATE = 0.001