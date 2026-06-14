import torch
import torch.nn as nn


class ElmanRNN(nn.Module):
    def __init__(self, dim, system='VanDerPol', hidden_dim=None, num_layers=1):
        super(ElmanRNN, self).__init__()
        ##############
        ##############

        # store basic info
        self.dim = dim
        self.system = system

        # choose hidden size (you can tune these)
        if hidden_dim is None:
            hidden_dim = 10

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # classic RNN cell with tanh nonlinearity and a linear readout
        self.rnn = nn.RNN(
            input_size=dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            nonlinearity="tanh",
            batch_first=False,
        )
        self.readout = nn.Linear(hidden_dim, dim)

        ##############
        ##############

    def forward(self, x, h0=None):
        ##############
        ##############

        # x is expected to have shape (n_timesteps, dim)
        # or (dim,) for a single state in testing mode
        squeezed = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeezed = True

        # add batch dimension for the RNN: (seq_len, batch, dim)
        x = x.unsqueeze(1)
        out, hn = self.rnn(x, h0)
        y = self.readout(out).squeeze(1)

        if squeezed:
            y = y.squeeze(0)

        ##############
        ##############
        return y, hn

    