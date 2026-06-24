import torch
import torch.nn.functional as F

def compute_loss(outputs, targets, hidden_states=None, predictions=None, inputs=None, 
                 condition="vanilla", lambda_reg=1e-4, mu_reg=0.1):
    """
    Computes the loss based on the experimental condition.
    - outputs: (batch, seq, output_size)
    - targets: (batch, seq) - typically only evaluated at the final timestep or decision period
    - predictions: (batch, seq, input_size) - predicted NEXT input
    """
    # 1. Vanilla Supervised (Cross Entropy on the last timestep for PoC)
    # Note: In real NeuroGym tasks, you mask this to the decision period.
    ce_loss = F.cross_entropy(outputs[:, -1, :], targets)
    
    total_loss = ce_loss
    
    # 2. Activity-regularized (Efficient Coding)
    if condition == "efficient":
        # lambda * mean(h^2) across all timesteps and batches
        activity_penalty = lambda_reg * torch.mean(hidden_states ** 2)
        total_loss += activity_penalty
        
    # 3. Predictive auxiliary (Predictive Coding)
    elif condition == "predictive":
        # mu * MSE(pred(t), u(t+1))
        # We predict the input at the next timestep. 
        # Exclude the last prediction since there is no t+1 input to compare it to.
        pred_t = predictions[:, :-1, :]
        actual_next_u = inputs[:, 1:, :]
        
        predictive_loss = mu_reg * F.mse_loss(pred_t, actual_next_u)
        total_loss += predictive_loss
        
    return total_loss