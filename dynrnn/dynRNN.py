import torch
import torch.nn as nn


class DynRNN(nn.Module):
    def __init__(self, dim, system='VanDerPol'):
        super(DynRNN, self).__init__()
        ##############
        ##############

        # store basic info
        self.dim = dim
        self.system = system

        # choose hidden size (you can tune these)
        if system == 'VanDerPol':
            hidden_dim = 300
        elif system == 'Lorenz':
            hidden_dim = 300

        # recurrent block: simple feedforward network with 1-2 hidden layers
        # tanh activations, final layer without activation
        self.recurrent_block = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim),  # output layer, no activation
        )

        ##############
        ##############

    def forward(self, x):
        ##############
        ##############

        # x is expected to have shape (n_timesteps, dim)
        # or (dim,) for a single state in testing mode
        squeezed = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeezed = True

        # apply the recurrent block timestep-wise (batch dimension = time)
        x = self.recurrent_block(x)

        if squeezed:
            x = x.squeeze(0)

        ##############
        ##############
        return x

    def calculate_jacobian(self, x):
        """" Calculates Jacobians along trajectory x.
        For every timestep the Jacobian is calculated as the derivative of the predictions wrt. inputs
        and has a shape (n_dim, n_dim) at every timestep.
        The final Jacobian (return variable) contains the concatenated Jacobians for all timesteps
        and has a shape (n_timesteps, n_dim, n_dim).

        Parameters:
        -----------
        x : torch.Tensor
            trajectory values, with shape (n_timesteps, n_dim)


        Return:
        -----------
        jacobian : torch.Tensor
            computed jacobians for the given trajectory
            shape = (n_timesteps, n_dim, n_dim)

        """
        jacobian = None
        ##############
        ##############

        assert x.dim() == 2, "x must have shape (n_timesteps, n_dim)"
        n_timesteps, n_dim = x.shape
        device = x.device
        dtype = x.dtype

        # storage for all Jacobians
        jacobian = torch.zeros(n_timesteps, n_dim, n_dim, device=device, dtype=dtype)

        # start from the first point of the trajectory as initial state
        # then always use the model's own prediction as the next state
        state = x[0].clone().detach().requires_grad_(True)

        for t in range(n_timesteps):

            # prediction of the next state from the current state
            with torch.enable_grad():
                pred = self.forward(state)  # shape (n_dim,)

                # compute Jacobian d(pred) / d(state)
                for i in range(n_dim):
                    grad_output = torch.zeros_like(pred)  # Partial derivative of 
                    # predicted dim i wrt to all previous state dims
                    grad_output[i] = 1.0 
                    # If dim=2, grad_output is [1,0] for i=0 and [0,1] for i=1
                    # This allows to select which output dimension to differentiate,
                    # i.e., which predicted state (y1(t) or y2(t)) to differentiate wrt. 
                    # the two input states (y1(t-1), y2(t-1)) hence, which row of 
                    # the Jacobian to compute.

                    grads = torch.autograd.grad(
                        outputs=pred, # vector, shape (n_dim,)
                        inputs=state, # vector, shape (n_dim,)
                        grad_outputs=grad_output,
                        retain_graph=(i < n_dim - 1),
                        create_graph=False,
                        allow_unused=False,
                    )[0]  # same shape as state: (n_dim,)

                    jacobian[t, i, :] = grads  # row i of the Jacobian at time t

            # roll the system forward: next state is the prediction
            state = pred.detach().requires_grad_(True)

        ##############
        ##############
        return jacobian
