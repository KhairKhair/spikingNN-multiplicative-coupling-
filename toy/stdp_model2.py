import torch
import torch.nn as nn
import norse
import norse.torch.functional.stdp as stdp_fn
from coupled_neuron.coupled_lif import CoupledLIFCell
import time

device = torch.device("cpu")

def measure_assignment_stability(old_assignments, new_assignments):
    """Return fraction of neurons whose assignment changed."""
    changes = (old_assignments != new_assignments).sum().item()
    return changes / len(old_assignments)

class SNNWithSTDP3(nn.Module):
    def __init__(self, input_size, hidden_size, num_labels, use_coupled, use_movingAvg, lam, eta):
        super(SNNWithSTDP3, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = 50
        self.encoder = norse.torch.module.encode.PoissonEncoder(seq_length=50)

        exc_params = norse.torch.functional.lif.LIFParameters(
            tau_syn_inv=torch.tensor(1/1e-2),
            tau_mem_inv=torch.tensor(1/2e-2),
            v_leak=torch.tensor(0.0),
            v_th=torch.tensor(0.5),
            v_reset=torch.tensor(0.0),
            method='super',
            alpha=torch.tensor(100.0)
        )

        self.lif = CoupledLIFCell(
            input_size=input_size,
            hidden_size=hidden_size,
            p=exc_params,
            use_coupled=use_coupled,
            use_movingAvg=use_movingAvg,
            lam=lam, eta=eta
        )

        self.num_labels = num_labels
        self.assignments = self.init_assignments()
        self.spike_count = torch.zeros(num_labels, hidden_size)
        self.label_count = torch.zeros(num_labels)

        self.criterion = nn.CrossEntropyLoss()

        # STDP parameters
        self.p_stdp = stdp_fn.STDPParameters(
            a_pre=torch.tensor(0.001, device=device),
            a_post=torch.tensor(0.001, device=device),
            tau_pre_inv=torch.tensor(1/20e-1, device=device),
            tau_post_inv=torch.tensor(1/20e-1, device=device),
            w_min=torch.tensor(0.0, device=device),
            w_max=torch.tensor(0.3, device=device),
            eta_plus=torch.tensor(1e-2, device=device),
            eta_minus=torch.tensor(1e-4, device=device),
            stdp_algorithm="additive",
            hardbound=False
        )
        self.state_stdp_input = stdp_fn.STDPState(
            t_pre=torch.zeros(self.hidden_size, self.input_size, device=device),
            t_post=torch.zeros(self.hidden_size, device=device)
        )
        self.state_stdp_recurrent = stdp_fn.STDPState(
            t_pre=torch.zeros(self.hidden_size, self.hidden_size, device=device),
            t_post=torch.zeros(self.hidden_size, device=device)
        )

        # For monitoring stats
        self.spike_history = []            # Will store average spike rate each epoch
        self.assignment_stability = []     # Will store fraction of changed assignments each epoch

    def init_assignments(self):
        self.start = True
        neurons_per_digit = self.hidden_size // self.num_labels
        assignment = []
        for digit in range(self.num_labels):
            assignment += [digit] * neurons_per_digit
        remaining = self.hidden_size - len(assignment)
        if remaining > 0:
            assignment += [self.num_labels - 1] * remaining
        assignment = torch.tensor(assignment, dtype=torch.long)
        return assignment

    def update_assignments(self):
        label_count_nonzero = torch.where(
            self.label_count > 0,
            self.label_count,
            torch.ones_like(self.label_count)
        )
        normalized_spike_count = self.spike_count / label_count_nonzero.unsqueeze(1)
        new_assignments = torch.argmax(normalized_spike_count, dim=0)
        spike_sum = self.spike_count.sum(dim=0)
        zero_spike_mask = spike_sum == 0

        updated_assignments = torch.where(
            zero_spike_mask,
            self.assignments,
            new_assignments
        ).detach()

        self.assignments = updated_assignments
        self.spike_count.zero_()
        self.label_count.zero_()

    def forward(self, x, label, is_test=False):
        with torch.no_grad():
            time_window, batch_size, _ = x.size()
            if label.dim() == 0:
                label = label.unsqueeze(0)
            labels_as_indices = label

            a = torch.nn.functional.one_hot(self.assignments, num_classes=self.num_labels).float()
            b = a.sum(dim=0).clamp(min=1)
            spike_counts_batch = torch.zeros(batch_size, self.hidden_size)
            state = None

            # Reinitialize STDP states each forward pass
            self.state_stdp_input = stdp_fn.STDPState(
                t_pre=torch.zeros(self.hidden_size, self.input_size),
                t_post=torch.zeros(self.hidden_size)
            )
            self.state_stdp_recurrent = stdp_fn.STDPState(
                t_pre=torch.zeros(self.hidden_size, self.hidden_size),
                t_post=torch.zeros(self.hidden_size)
            )

            outputs = []
            batch_energy = 0.0
            total_spikes = 0.0  # Keep track of total spikes

            for t in range(time_window):
                pre_spikes = x[t]
                post_spikes, state = self.lif(pre_spikes, state)
                outputs.append(post_spikes)

                # Count spikes
                spike_counts_batch += post_spikes
                total_spikes += post_spikes.sum().item()

                # Update label counts for assignment

                if not is_test:
                    for batch in range(batch_size):
                        lbl = labels_as_indices[batch]
                        self.label_count[lbl] += 1
                        self.spike_count[lbl] += post_spikes[batch]
                    # STDP updates
                    self.lif.input_weights.data, self.state_stdp_input = stdp_fn.stdp_step_linear(
                        pre_spikes,
                        post_spikes,
                        self.lif.input_weights.data,
                        self.state_stdp_input,
                        self.p_stdp,
                        dt=0.001
                    )
                    if t > 0:
                        pre_spikes_recurrent = outputs[t - 1]
                    else:
                        pre_spikes_recurrent = torch.zeros_like(post_spikes)

                    self.lif.recurrent_weights.data, self.state_stdp_recurrent = stdp_fn.stdp_step_linear(
                        pre_spikes_recurrent,
                        post_spikes,
                        self.lif.recurrent_weights.data,
                        self.state_stdp_recurrent,
                        self.p_stdp,
                        dt=0.001
                    )
                else:
                    # Energy calculation if is_test
                    E_t = -0.5 * torch.einsum('bi,ij,bj->b',
                        post_spikes,
                        self.lif.recurrent_weights,
                        post_spikes)
                    batch_energy += E_t.sum()

            # Compute logit-like outputs for classification
            spikes_per_batch = spike_counts_batch @ a
            output_logits = spikes_per_batch / (time_window * b)

            # Return spike info to compute average spike rate in _train
            return output_logits, batch_energy, total_spikes

    def _train(self, num_epochs, train_loader, test_loader):
        self.acc_history = []
        self.energy_history = []
        self.a = []
        self.b = []

        for epoch in range(num_epochs):
            start = time.time()
            for batch_input, batch_target in train_loader:
                batch_size = batch_input.size(0)
                batch_input = batch_input.permute(1,0,2)   # (T, B, input_size)
                batch_target = batch_target.squeeze(0)

                _, _, batch_total_spikes = self(batch_input, batch_target, is_test=False)

            self.update_assignments()

            # Evaluate on test set
            acc, total_e = self._test(test_loader)
            self.acc_history.append(acc)
            self.energy_history.append(total_e.item())
            print("Time Taken: ", time.time()-start)


    def _test(self, test_loader):
        self.eval()
        total_correct = 0
        total_samples = 0
        total_energy = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                batch_size = inputs.size(0)
                inputs = inputs.permute(1,0,2)
                outputs, energy, _ = self(inputs, targets, is_test=True)  # returns output_logits, energy, spikes
                _, predicted = torch.max(outputs, dim=1)
                total_correct += (predicted == targets).sum().item()
                total_samples += batch_size
                total_energy += energy

        accuracy = total_correct / total_samples
        print(f"Test Accuracy: {accuracy:.4f}, Energy: {total_energy:.2f}", flush=True)
        return accuracy, total_energy


