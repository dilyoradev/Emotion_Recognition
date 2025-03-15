from data_processing import load_data
from model import create_model, loss_fn, optimizer
from train import train_model
from test import test_model
from save_load import save_model, load_model
from config import DEVICE, LEARNING_RATE, TRAIN_DIR, TEST_DIR, BATCH_SIZE, NUM_EPOCHS
import torch

print(torch.backends.mps.is_available())  # Should return True
print(torch.backends.mps.is_built())      # Should return True
   
def main():
    # Load data
    train_dataloader, test_dataloader  = load_data(TRAIN_DIR, TEST_DIR, BATCH_SIZE)
    
    # Initialize model
    model = create_model()
    
    # Initialize optimizer with model parameters
    optimizer_instance = optimizer(model.parameters(), lr=LEARNING_RATE)
    # Train the model
    print("Starting training...")
    model.to("mps")
    train_losses, train_accuracies = train_model(model, train_dataloader, optimizer_instance, NUM_EPOCHS)
    
    # Save the model
    save_model(model)
    load_model(model)
    
    # Test the model
    test_loss, test_acc = test_model(model, test_dataloader, loss_fn)
    print("Training and evaluation completed!")
    
if __name__ == "__main__":
    main()