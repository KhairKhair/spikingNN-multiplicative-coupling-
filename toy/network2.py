import torch
import torch.nn as nn
import norse
from coupled_neuron.coupled_lif import CoupledLIFCell
import numpy as np
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class SNNToy(nn.Module):
    def __init__(self, input_size, hidden_size, num_labels, learning_rate, use_coupled, use_movingAvg, lam, eta):
        super(SNNToy, self).__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
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


        self.lif.recurrent_weights.requires_grad = True
        self.lif.input_weights.requires_grad = True

        self.readout = nn.Linear(hidden_size, num_labels)
        self.readout.weight.requires_grad = True
        self.readout.bias.requires_grad = True

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        self.loss_history = [] 
        self.test_history = []
        self.acc_history = []


    def forward(self, x, is_test=False): 

        state = None

        time_window, batch_size, num_neurons = x.size()
        state = None
        outputs = torch.zeros(batch_size, self.num_labels, device=device)
        batch_energy = 0.0

        for t in range(time_window):
            out, state = self.lif(x[t], state)  

            if is_test:
                E_t = -0.5 * torch.einsum('bi,ij,bj->b', 
                    out, 
                    self.lif.recurrent_weights, 
                    out)
                batch_energy += E_t.sum()

            out = self.readout(out)  
            #out = torch.nn.functional.sigmoid(out)
            outputs += out



        return outputs.float()/time_window, batch_energy


    def _train(self, num_epochs, x, test=False):
        self.acc_history = []
        self.energy_history = []
        
        num_samples = len(x.dataset)
        batch_size = 1
        for batch in x:
            batch_size = batch[0].size(0)
            break
        
        for epoch in range(num_epochs):
            start_time = time.time()
            epoch_loss = 0
            for batch_input, batch_target in x:
                batch_input = batch_input.permute(1,0,2).to(device)
                batch_target = batch_target.to(device)
                batch_target = batch_target.to(device)
                
                self.optimizer.zero_grad()
                output, _ = self(batch_input, is_test=False)
                
                loss = self.criterion(output, batch_target)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss
                    
            acc, total_e = self.test(test, device)
            self.acc_history.append(acc)
            self.energy_history.append(total_e.item())
            print("Time taken: ", time.time()-start_time, flush=True)



    def test(model, test_loader, device):
        model.eval() 
        total_loss = 0
        correct = 0
        #criterion = torch.nn.CrossEntropyLoss()
        energy_total = 0.0

        with torch.no_grad():  
            for batch_input, batch_target in test_loader:
                batch_input = batch_input.permute(1,0,2).to(device)
                batch_target = batch_target.to(device)
    
                # Forward pass
                output, e = model(batch_input, is_test=True)
                energy_total += e
    
                # Calculate loss
                #loss = criterion(output, batch_target)
                #total_loss += loss.item()
    
                preds = output.argmax(dim=1)  
                correct += (preds == batch_target).sum().item()
    
        avg_loss = total_loss / len(test_loader)
        accuracy = correct / (len(test_loader.dataset))
    
        print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}, Energy: {energy_total}", flush=True)
        return accuracy, energy_total
    
    
    
    

