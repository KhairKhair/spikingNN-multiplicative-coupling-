# Import necessary libraries for neural network training and data processing
from jax import numpy as jnp, random
import sys, time
from dcsnn_model import DC_SNN as Model  # Ensure this path is correct
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Set training parameters
n_epochs = 40  # Total number of passes through the dataset (epochs)
n_samples = 200  # Number of training samples to use (will be overridden by command line args)
# Define file paths for MNIST dataset
dataX = "../data/mnist/trainX.npy"
dataY = "../data/mnist/trainY.npy"
testX = "../data/mnist/testX.npy"
testY = "../data/mnist/testY.npy"
verbosity = 1  # Verbosity level (0 - minimal, 1 - detailed)

print("started", flush=True)

# Helper function to convert string arguments to boolean values
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Set up command line argument parser for configurable training parameters
parser = argparse.ArgumentParser(description="SNN with STDP Parameters")
parser.add_argument('--seed', type=int, required=True, help='Random seed')
parser.add_argument('--use_coupled', type=str2bool, required=True, help='Use coupled LIF cell (True/False)')
parser.add_argument('--max_samples', type=int, required=True, help='Ufwie coupled LIF cell (True/False)')
parser.add_argument('--hidden_size', type=int, required=True, help='Ufwie coupled LIF cell (Truefiewji/False)')
args = parser.parse_args()
n_samples = args.max_samples  # Override default n_samples with command line argument

print(args.seed, args.use_coupled, args.max_samples, args.hidden_size, flush=True)

# Load training dataset from numpy files
_X_train = jnp.load(dataX)
_Y_train = jnp.load(dataY)

# Convert one-hot encoded labels to scalar labels if necessary
if _Y_train.ndim > 1 and _Y_train.shape[1] > 1:
    _Y_train = jnp.argmax(_Y_train, axis=1)
elif _Y_train.ndim == 1:
    _Y_train = jnp.squeeze(_Y_train)
else:
    raise ValueError("Unsupported training label format. Labels should be either scalar or one-hot encoded.")

# Limit the dataset size if n_samples is specified
if n_samples > 0:
    _X_train = _X_train[:n_samples, :]
    _Y_train = _Y_train[:n_samples]

n_batches_train = _X_train.shape[0]  # Number of training samples
patch_shape = (28, 28)  # MNIST image dimensions

# Simulation parameters
T = 200  # Total simulation time steps
dt =  1  # Time step size

print("--- Building Model ---", flush=True)
# Initialize random key for reproducible results
dkey = random.PRNGKey(args.seed)
# Create the DC-SNN model with specified parameters
model = Model(dkey=dkey, in_dim=patch_shape[0] * patch_shape[1], T=T, dt=dt, use_coupled=args.use_coupled, hid_dim=args.hidden_size, use_movingAvg=True, lem=0.5, eta=0.5)
print("--- Starting Simulation ---", flush=True) 

#model.save_to_disk()

# Function to monitor and save weight statistics during training
def monitor_weights(epoch, model):
    weights_w1 = model.W1.weights.value
    if hasattr(model, 'W1re'):
        weights_w1re = model.W1re.weights.value
    else:
        weights_w1re = None
    
    # Print weight statistics for monitoring training progress
    print(f"\n--- Weight Stats (Epoch {epoch}) ---", flush=True)
    print(f"W1 - min: {jnp.min(weights_w1)}, max: {jnp.max(weights_w1)}, mean: {jnp.mean(weights_w1)}, norm: {jnp.linalg.norm(weights_w1)}", flush=True)
    if weights_w1re is not None:
        print(f"W1re - min: {jnp.min(weights_w1re)}, max: {jnp.max(weights_w1re)}, mean: {jnp.mean(weights_w1re)}, norm: {jnp.linalg.norm(weights_w1re)}", flush=True)
    
    # Save weight matrices to disk for later analysis
    jnp.save(f"{model.exp_dir}/weights_w1_epoch{epoch}.npy", weights_w1)
    if weights_w1re is not None:
        jnp.save(f"{model.exp_dir}/weights_w1re_epoch{epoch}.npy", weights_w1re)

# Function to visualize spike trains as raster plots
def visualize_spike_train(spikes, epoch, index=0):
    plt.figure(figsize=(10, 5))
    spike_train = jnp.squeeze(spikes[index])
    plt.imshow(spike_train.T, aspect='auto', cmap='binary')
    plt.colorbar(label='Spike')
    plt.xlabel('Time Step')
    plt.ylabel('Neuron Index')
    plt.title(f'Spike Raster Plot (Epoch {epoch}, Sample {index})')
    plt.savefig(f"spike_raster_epoch{epoch}_sample{index}.png")
    plt.close()

# Function to compute and display spike statistics
def get_spike_stats(spikes, epoch):
    total_spikes = jnp.sum(jnp.squeeze(spikes), axis=0)  # Total spikes per neuron
    zero_spikes = jnp.sum(total_spikes == 0)  # Count neurons with no spikes
    print(f"Epoch {epoch} - Total spikes per neuron (non-zero spikes): {total_spikes}", flush=True)
    print(f"Epoch {epoch} - Neurons with zero spikes: {zero_spikes}/{len(total_spikes)}", flush=True)

# Function to initialize neuron-to-label assignments for classification
def init_assignments(num_labels, hidden_size):
    neurons_per_label = hidden_size // num_labels  # Evenly distribute neurons across labels
    assignment = []
    for label in range(num_labels):
        assignment += [label] * neurons_per_label
    # Assign remaining neurons to the last label
    remaining = hidden_size - len(assignment)
    if remaining > 0:
        assignment += [num_labels - 1] * remaining
    assignment = jnp.array(assignment)
    return assignment

# Function to update neuron assignments based on spike activity patterns
def update_assignments(spike_count, label_count, assignments):
    # Normalize spike counts by label frequency to avoid bias
    label_count_nonzero = jnp.where(label_count > 0, label_count, jnp.ones_like(label_count))
    normalized_spike_count = spike_count / label_count_nonzero[:, None]
    # Assign each neuron to the label it spikes most for
    new_assignments = jnp.argmax(normalized_spike_count, axis=0)
    # Keep original assignments for neurons that never spike
    spike_sum = spike_count.sum(axis=0)
    zero_spike_mask = spike_sum == 0
    updated_assignments = jnp.where(zero_spike_mask, assignments, new_assignments)
    return updated_assignments

# Function to evaluate model accuracy on test data
def evaluate_accuracy(model, neuron_labels, _X_test, _Y_test, verbosity=0):
    correct_predictions = 0
    total_samples = _X_test.shape[0]

    # Process each test sample individually
    for j in range(total_samples):
        Xb = _X_test[j:j + 1, :]  # Single sample batch
        Yb = _Y_test[j]
        
        # Get spike train without adapting synapses (inference mode)
        _S = model.process(obs=Xb, adapt_synapses=False, collect_spike_train=True)
        rates = jnp.sum(jnp.squeeze(_S), axis=0)  # Sum spikes across time for each neuron
        predicted_neuron = jnp.argmax(rates).item()  # Most active neuron
        predicted_label = int(neuron_labels[predicted_neuron])  # Label of most active neuron

        if predicted_label == int(Yb):
            correct_predictions += 1

    accuracy = (correct_predictions / total_samples) * 100
    return accuracy

# Function to compute normalized Boltzmann energy for the network
def compute_boltzmann_energy_normalized(model, spike_count, weight_scale=1e-3, spike_scale=1e-3):
    W_re = model.W1re.weights.value * weight_scale  # Scale weights
    S = jnp.sum(spike_count, axis=0) * spike_scale  # Scale spike counts
    energy = -jnp.dot(S, jnp.dot(W_re, S))  # Compute energy as -S^T * W * S
    return energy

# Load test dataset
_X_test = jnp.load(testX)
_Y_test = jnp.load(testY)

# Convert test labels to scalar format if necessary
if _Y_test.ndim > 1 and _Y_test.shape[1] > 1:
    _Y_test = jnp.argmax(_Y_test, axis=1)
elif _Y_test.ndim == 1:
    _Y_test = jnp.squeeze(_Y_test)
else:
    raise ValueError("Unsupported test label format. Labels should be either scalar or one-hot encoded.")

# Limit test dataset size to match training data
if n_samples > 0:
    _X_test = _X_test[:n_samples, :]
    _Y_test = _Y_test[:n_samples]

n_batches_test = _X_test.shape[0]

# Initialize variables for tracking neuron assignments and spike statistics
num_labels = 10  # Assuming 10 labels for MNIST
hidden_size = model.z1e.n_units
assignments = init_assignments(num_labels, hidden_size)  # Initial neuron-to-label mapping
spike_count = jnp.zeros((num_labels, hidden_size))  # Track spikes per label per neuron
label_count = jnp.zeros(num_labels)  # Count occurrences of each label

# Lists to store training metrics across epochs
all_acc = []  # Store accuracy for each epoch
all_e = []    # Store energy for each epoch

# Main training loop
for epoch in range(1, n_epochs + 1):
    print(f"\n--- Epoch {epoch}/{n_epochs} ---", flush=True)
    start = time.time()
    # Generate new random permutation for this epoch
    dkey, *subkeys = random.split(dkey, 2)
    ptrs = random.permutation(subkeys[0], _X_train.shape[0])
    X = _X_train[ptrs, :]  # Shuffled training data
    Y = _Y_train[ptrs]     # Corresponding shuffled labels
    
    # Process each training sample
    for j in range(n_batches_train):
        idx = j
        Xb = X[idx: idx + 1, :]  # Single sample batch
        Yb = Y[idx: idx + 1]     # Corresponding label
        
        # Process sample through network with STDP learning enabled
        spikes = model.process(obs=Xb, adapt_synapses=True, collect_spike_train=True)

        # Update spike statistics for assignment learning
        spike_sum = jnp.sum(spikes, axis=(0, 1))  # Shape: (hidden_size,)
        label = int(Yb[0].item())  # Since batch_size=1, Yb[0] is the label
        label_count = label_count.at[label].add(1)  # Increment label count
        spike_count = spike_count.at[label].add(spike_sum)  # Add spike counts

    print(f"training took {time.time()-start}", flush=True)
    start_test = time.time()
    
    e = compute_boltzmann_energy_normalized(model, spike_count)
    print("energy normal: ", e)
    
    # Update neuron assignments based on spike patterns
    assignments = update_assignments(spike_count, label_count, assignments)
    
    # Evaluate model accuracy on test set
    accuracy = evaluate_accuracy(model, assignments, _X_test, _Y_test, verbosity=verbosity)
    all_acc.append(accuracy)  # Store accuracy for this epoch
    all_e.append(e)           # Store energy for this epoch
    print(f"Accuracy after epoch {epoch}: {accuracy:.2f}%", flush=True)
    print("testing took", time.time()-start_test, flush=True)

# Print final training results
print("Final accuracies: ", all_acc)
print("Final Energies: ", all_e)
