import torch
import torch.nn as nn
import norse
from coupled_neuron.coupled_lif import CoupledLIFCell
import numpy as np
import time
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class SNNMnist(nn.Module):
    def __init__(self, input_size, hidden_size, num_labels, learning_rate, use_coupled, use_movingAvg):
        super(SNNMnist, self).__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.encoder = norse.torch.module.encode.PoissonEncoder(20)


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
             lam=0.5, eta=0.5
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

        for t in range(time_window):
            out, state = self.lif(x[t], state)  

            out = self.readout(out)  
            outputs += out



        return outputs.float()/time_window, 0.01


    def _train(self, num_epochs, x, test=False):
        self.acc_history = []
        
        num_samples = len(x.dataset)
        batch_size = 1
        for batch in x:
            batch_size = batch[0].size(0)
            break
        
        acc, total_e = self.test(test, device)
        self.acc_history.append(acc)
        print("Test Accuracy of epoch 0 ", " is: ", acc)
        for epoch in range(num_epochs+1):
            start_time = time.time()
            epoch_loss = 0
            for batch_input, batch_target in tqdm(x):
                batch_input = self.encoder(batch_input).to(device)
                batch_target = batch_target.to(device)
                batch_target = batch_target.to(device)
                
                self.optimizer.zero_grad()
                output, _ = self(batch_input, is_test=False)
                
                loss = self.criterion(output, batch_target)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss

                del batch_input, batch_target, output, loss
                torch.cuda.empty_cache()  # Optional
                    
            acc, total_e = self.test(test, device)
            self.acc_history.append(acc)
            print("Time taken: ", time.time()-start_time, flush=True)
            print("Test Accuracy of epoch ", epoch, " is: ", acc)



    def test(self, test_loader, device):
        self.eval() 
        total_loss = 0
        correct = 0
        #criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():  
            for batch_input, batch_target in test_loader:
                batch_input = self.encoder(batch_input).to(device)
                batch_target = batch_target.to(device)
    
                # Forward pass
                output, e = self(batch_input, is_test=True)
    
                # Calculate loss
                #loss = criterion(output, batch_target)
                #total_loss += loss.item()
    
                preds = output.argmax(dim=1)  
                correct += (preds == batch_target).sum().item()
    
        avg_loss = total_loss / len(test_loader)
        accuracy = correct / (len(test_loader.dataset))
    
        return accuracy, 0.0
    
    
    
    

