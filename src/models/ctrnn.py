import torch
import torch.nn as nn

class CTRNN(nn.Module):
    def __init__(self, input_size, hidden_size=80, output_size=3, dt=20, tau=100):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Euler integration constant
        self.alpha = dt / tau
        
        # Network layers
        self.i2h = nn.Linear(input_size, hidden_size, bias=True)
        self.h2h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.h2o = nn.Linear(hidden_size, output_size, bias=True)
        
        # Predictive head for the Predictive Coding condition
        self.h2pred = nn.Linear(hidden_size, input_size, bias=True)

        self.init_weights()

    def init_weights(self):
        # W_rec orthogonal (gain 1.0)
        nn.init.orthogonal_(self.h2h.weight, gain=1.0)
        # W_in, W_out, W_pred Xavier uniform
        nn.init.xavier_uniform_(self.i2h.weight)
        nn.init.xavier_uniform_(self.h2o.weight)
        nn.init.xavier_uniform_(self.h2pred.weight)
        # Biases to 0
        nn.init.zeros_(self.i2h.bias)
        nn.init.zeros_(self.h2o.bias)
        nn.init.zeros_(self.h2pred.bias)

    def forward(self, x, return_dynamics=False):
        # x shape: (batch_size, sequence_length, input_size)
        batch_size, seq_len, _ = x.size()
        
        # Initialize hidden state at zero
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        outputs = []
        predictions = []
        hidden_states = []
        
        for t in range(seq_len):
            u_t = x[:, t, :]
            
            # Euler integration step
            dx = -h + self.h2h(torch.tanh(h)) + self.i2h(u_t)
            h = h + self.alpha * dx
            
            # Compute outputs
            y_t = self.h2o(h)
            pred_t = self.h2pred(h)
            
            outputs.append(y_t)
            predictions.append(pred_t)
            if return_dynamics:
                hidden_states.append(h)
                
        outputs = torch.stack(outputs, dim=1)
        predictions = torch.stack(predictions, dim=1)
        
        if return_dynamics:
            hidden_states = torch.stack(hidden_states, dim=1)
            return outputs, predictions, hidden_states
            
        return outputs, predictions