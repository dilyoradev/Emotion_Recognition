from imports import *
from model import model
from config import DEVICE

FILENAME = "model.pth"
def save_model(model):
    """Saves the trained model weights to a file."""
    torch.save(model.state_dict(), FILENAME)
    print(f"Model saved successfully as {FILENAME}")
    
def load_model(model):
    """Loads model weights from the saved file."""
    model.load_state_dict(torch.load(FILENAME, map_location=DEVICE))
    model.eval()
    print(f"Model loaded successfully from {FILENAME}")