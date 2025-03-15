from imports import *

def create_model():
    """Creates and returns the ResNet50 model with a modified classifier."""
    # model setup
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device)

# Create a model instance
model = create_model()

# Initialize loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam
