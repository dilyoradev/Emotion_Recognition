import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from tqdm.auto import tqdm
from timeit import default_timer as timer
