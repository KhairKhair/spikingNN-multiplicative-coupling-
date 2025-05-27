import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from network2 import SNNToy
import argparse
from helper_plot_toy import set_seed, NumpyDataset
from stdp_model2 import SNNWithSTDP3 as model
import matplotlib.pyplot as plt

n_epochs = 50  # Total number of passes through the dataset (epochs)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description="SNN with STDP Parameters")
parser.add_argument('--seed', type=int, required=True, help='Random seed')
parser.add_argument('--use_coupled', type=str2bool, required=True, help='Use coupled LIF cell (True/False)')
parser.add_argument('--hidden', type=int, required=True, help='hidden')
parser.add_argument('--lam', type=float, required=True, help='hidden')
parser.add_argument('--eta', type=float, required=True, help='hidden')
parser.add_argument('--batch', type=int, required=True, help='hidden')

args = parser.parse_args()
trainX = np.load("./new_toy/trainX.npy")
trainY = np.load("./new_toy/trainY.npy")
testX = np.load("./new_toy/testX.npy")
testY = np.load("./new_toy/testY.npy")

train_dataset = NumpyDataset(trainX, trainY, max_samples=100)
test_dataset = NumpyDataset(testX, testY, max_samples=300)

set_seed(args.seed)
train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_size = args.hidden
num_labels = 2
num_epochs = 100


#net = SNNToy(100, hidden_size, num_labels, use_coupled=args.use_coupled, use_movingAvg=args.use_coupled, learning_rate=0.1, lam=args.lam, eta=args.eta).to(device)
#net._train(num_epochs, train_loader, test_loader)

net = model(100, hidden_size, num_labels, args.use_coupled, args.use_coupled, args.lam, args.eta)
net._train(num_epochs, train_loader, test_loader)

print("Final accuracies:", net.acc_history, flush=True)
print("Final energies:", net.energy_history, flush=True)

