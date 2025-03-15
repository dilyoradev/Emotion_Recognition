from imports import *
from data_processing import load_data
from model import create_model, loss_fn, optimizer
from config import DEVICE, LEARNING_RATE, TRAIN_DIR, TEST_DIR, BATCH_SIZE
from time import time  

# Load Data
train_dataloader, _ = load_data(TRAIN_DIR, TEST_DIR, BATCH_SIZE)

# Initialize Model and Optimizer
model = create_model()
optimizer = optimizer(model.parameters(), LEARNING_RATE) 

def train_model(model, train_dataloader, optimizer, NUM_EPOCHS):
    """Trains the model and returns loss and accuracy"""
    # Lists to store training and test metrics
    train_losses = []
    train_accuracies = []
    
    train_time_start = time()

    for epoch in tqdm(range(NUM_EPOCHS)):
        train_loss = 0.0
        train_correct = 0
        total_train_samples = 0

        # Training loop
        model.train()
        for images, labels in train_dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track loss and compute accuracy
            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == labels).sum().item()
            total_train_samples += labels.size(0)

        # Compute average loss and accuracy
        train_loss /= len(train_dataloader)
        train_acc = train_correct / total_train_samples * 100

        # Store train metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {train_loss:.4f} - Accuracy: {train_acc:.2f}%")
        
    train_time_end = time()
    print(f"Training completed in {train_time_end - train_time_start:.2f} seconds")
    
    return train_losses, train_accuracies
