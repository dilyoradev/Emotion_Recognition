from imports import *

def load_data(TRAIN_DIR: str,
              TEST_DIR: str,
              BATCH_SIZE):
    """Loads train and test dataset with transformations(image augmentation) and returns DataLoaders"""
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomRotation(10),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                                   ])
    train_data = datasets.ImageFolder(TRAIN_DIR, transform=transform)
    test_data = datasets.ImageFolder(TEST_DIR, transform=transform)
    class_names = train_data.classes
    train_dataloader = DataLoader(train_data,
                                BATCH_SIZE,
                                shuffle=True,
                                num_workers=4,
                                pin_memory=True)
    test_dataloader = DataLoader(test_data,
                            BATCH_SIZE,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True)
    return train_dataloader, test_dataloader
    
    