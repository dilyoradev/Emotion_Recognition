from imports import *
from data_processing import load_data
from model import create_model, loss_fn
from config import DEVICE, TRAIN_DIR, TEST_DIR, BATCH_SIZE
import torch

# Load Data
_, test_dataloader = load_data(TRAIN_DIR, TEST_DIR, BATCH_SIZE)

# Initialize Model
model = create_model()

def test_model(model, test_dataloader, loss_fn):
    """Evaluates the model on the test dataset."""
    test_loss = 0.0
    test_correct = 0
    total_test_samples = 0

    model.eval()  
    with torch.no_grad():  
        for X_test, y_test in test_dataloader:
            X_test, y_test = X_test.to(DEVICE), y_test.to(DEVICE)
            test_output = model(X_test)
            loss = loss_fn(test_output, y_test)
            test_loss += loss.item()

            # Compute test accuracy
            test_preds = torch.argmax(test_output, dim=1)
            test_correct += (test_preds == y_test).sum().item()
            total_test_samples += y_test.size(0)
            
    # Compute final test loss and accuracy
    test_loss /= len(test_dataloader)
    test_acc = test_correct / total_test_samples * 100
    
    print(f"Test Loss: {test_loss:.4f} - Test Accuracy: {test_acc:.2f}%")

    return test_loss, test_acc  
  