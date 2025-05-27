import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from network2 import SNNMnist 
import time

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description="Train SNN on MNIST")
parser.add_argument('--seed', type=int, required=True, help='Random seed')
parser.add_argument('--use_coupled', type=str2bool, required=True, help='Use coupled LIF cell (True/False)')
parser.add_argument('--hidden', type=int, required=True, help='Hidden layer size for the SNN')
args = parser.parse_args()

# Set random seed for reproducibility
set_seed(args.seed)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transform to convert MNIST images to tensor and flatten them
transform = transforms.Compose([
    transforms.ToTensor(),
    # Flatten 28x28 images into 784 vector
    transforms.Lambda(lambda x: x.view(-1)) 
])

# Load MNIST dataset
train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)

batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

input_size = 28 * 28  # MNIST images flattened
num_labels = 10       # 10 digits (0-9)
hidden_size = args.hidden

net = SNNMnist(input_size, hidden_size, num_labels, learning_rate=0.1,
                use_coupled=args.use_coupled, use_movingAvg=False)
net.to(device)

# Train the model
net._train(28, train_loader, test_loader)

print("Final accuracies:", net.acc_history)
print("Final energies:", net.energy_history)

