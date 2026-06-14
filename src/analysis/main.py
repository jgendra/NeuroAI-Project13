import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from RNN import ElmanRNN

# Define output directory
dir = "src/analysis/results/"

# hyperparameters
BATCH_SIZE = 1   # not really used here, since we train on a single trajectory
EPOCHS = 2500   
LEARNING_RATE = 5e-3

# Choose dynamical system
system = 'Sinusoid'


def r2_score(y_pred, y_true):
    """
    Coefficient of determination (R²) between predictions and targets.

    R² measures the fraction of variance in y_true that is explained by
    y_pred, and is the standard 'accuracy' analog for regression tasks:
        R² = 1  → perfect prediction
        R² = 0  → model is no better than predicting the mean of y_true
        R² < 0  → model is worse than predicting the mean

    We use R² rather than normalised MSE because it is scale-free and has a
    natural [−∞, 1] range with a meaningful zero point, making curves from
    different runs directly comparable.

    Parameters
    ----------
    y_pred : torch.Tensor  (any shape)
    y_true : torch.Tensor  (same shape as y_pred)

    Returns
    -------
    float
    """
    # total variance of the targets (denominator)
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    # residual variance left unexplained by the predictions (numerator)
    ss_res = ((y_true - y_pred) ** 2).sum()
    # R² = 1 − (unexplained / total); clamp denominator to avoid div-by-zero
    return (1.0 - ss_res / ss_tot.clamp(min=1e-8)).item()


if __name__ == "__main__":

    # Generate a simple sinusoid trajectory to test the PID metrics. Size (300, 1)
    n_timesteps = 300
    period = 50.0
    t = torch.linspace(0, n_timesteps - 1, n_timesteps)
    y_true = torch.sin(2 * torch.pi * t / period).unsqueeze(1)
    # print(y_true.shape)

    n_timesteps, dim = y_true.shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_true = y_true.to(device)

    # Create RNN
    model = ElmanRNN(dim=dim, system=system).to(device)

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # create the optimizer
    loss_function = nn.MSELoss()  # loss function (MSE)

    # For tracking over epochs
    training_losses = []
    training_r2s = []          # R² at each epoch (teacher-forced predictions vs targets)

    # ===== TRAINING LOOP =====
    for epoch in range(EPOCHS):
        model.train()
        if epoch == 1000:
            for param_group in optimizer.param_groups:
                param_group["lr"] = 1e-3
        if epoch == 2000:
            for param_group in optimizer.param_groups:
                param_group["lr"] = 5e-4
        optimizer.zero_grad()

        # Teacher forcing over the full sequence: predict x_{t+1} from x_t
        x_in = y_true[:-1]
        target = y_true[1:]
        pred, _ = model(x_in)
        loss = loss_function(pred, target)

        # Backpropagate once after summing over all timesteps
        loss.backward()
        optimizer.step()

        # Track loss and R² for this epoch.
        # R² is computed on the same teacher-forced predictions used for the
        # loss, so it reflects how well the model fits the one-step-ahead
        # prediction task it is actually being trained on.
        training_losses.append(loss.item())
        training_r2s.append(r2_score(pred.detach(), target))

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1}/{EPOCHS}, "
                f"training loss = {loss.item():.6e}, "
                f"training R² = {training_r2s[-1]:.4f}"
            )

    # Save trained RNN
    torch.save(model.state_dict(), f"{dir}RNN100_{system}.pt")
    print(f"Trained RNN saved to {dir}RNN100_{system}.pt")


    # ===== EVALUATION: AUTOREGRESSIVE ROLLOUT =====
    model.eval()
    with torch.no_grad():
        y_pred = torch.zeros_like(y_true)
        # initial condition: first value in dataset
        y_pred[0] = y_true[0]

        # always feed prediction at n-1 as input to predict state at n
        h = None
        for t in range(n_timesteps - 1):
            step_pred, h = model(y_pred[t], h0=h)
            y_pred[t+1] = step_pred

        # Evaluation loss and R² on the autoregressive rollout.
        # This is harder than teacher-forced training: the model must sustain
        # the oscillation from its own outputs, so errors accumulate over time.
        mse_eval = loss_function(y_pred, y_true).item()
        r2_eval  = r2_score(y_pred, y_true)
        print(f"Evaluation MSE over full trajectory: {mse_eval:.6e}")
        print(f"Evaluation R²  over full trajectory: {r2_eval:.4f}")


    # ===== COLLECT HIDDEN STATES (for testing the PID metrics) =====
    model.eval()
    with torch.no_grad():
        # full hidden-state trajectory under teacher forcing over the sinusoid
        x_seq = y_true.unsqueeze(1)            # (seq_len, batch=1, dim)
        h_seq, _ = model.rnn(x_seq)            # (seq_len, batch=1, hidden_dim)
        h_seq = h_seq.squeeze(1)               # (seq_len, hidden_dim)
        torch.save(h_seq.cpu(), f"{dir}hidden100_{system}.pt")
        print(f"Hidden states {tuple(h_seq.shape)} saved to {dir}hidden100_{system}.pt")


    # ===== FIGURE 1: TRAINING CURVES (loss + R²) =====
    # Both metrics are plotted against epoch on a shared x-axis.
    # Loss (left y-axis) uses a log scale so the steep early drop and the
    # slow late convergence are both visible. R² (right y-axis) is linear.
    epochs_axis = range(1, EPOCHS + 1)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Left axis: MSE loss (log scale)
    color_loss = 'C0'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE loss (log scale)', color=color_loss)
    ax1.semilogy(epochs_axis, training_losses, color=color_loss,
                 linewidth=1.2, label='Training loss')
    ax1.tick_params(axis='y', labelcolor=color_loss)

    # Right axis: R²
    ax2 = ax1.twinx()
    color_r2 = 'C1'
    ax2.set_ylabel('R²', color=color_r2)
    ax2.plot(epochs_axis, training_r2s, color=color_r2,
             linewidth=1.2, label='Training R²')
    ax2.tick_params(axis='y', labelcolor=color_r2)
    ax2.set_ylim(-0.05, 1.05)          # R² lives in (−∞, 1]; clip low outliers

    # Combined legend for both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    plt.title(f'Training curves — {system} RNN')
    plt.tight_layout()
    plt.savefig(f"{dir}sinusoid100_training_curves_{system}.png", dpi=300)
    plt.show()


    # ===== FIGURE 2: TRUE VS PREDICTED TRAJECTORY =====
    y_true_np = y_true.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()
    time = range(n_timesteps)

    # Plot for Sinusoid system
    if system == 'Sinusoid':
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(time, y_true_np[:, 0], label='y_true', c='C0')
        ax.plot(time, y_pred_np[:, 0], '--', label='y_predicted', c='C1')

        # Evaluation metrics shown as a text box in the upper-right corner.
        # Using a semi-transparent box so the label is readable over the curve.
        metrics_text = (
            f"Eval MSE = {mse_eval:.4e}\n"
            f"Eval R²  = {r2_eval:.4f}"
        )
        ax.text(
            0.98, 0.97,                 # upper-right in axes-fraction coordinates
            metrics_text,
            transform=ax.transAxes,     # interpret x,y as fractions of the axes
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      alpha=0.75, edgecolor='0.7'),
        )

        ax.set_ylabel('y')
        ax.set_title(f'Sinusoid trajectory: true vs predicted ({system} RNN)')
        ax.set_xlabel('Time (s)')
        ax.legend(loc='upper left')

        plt.tight_layout()
        plt.savefig(f"{dir}sinusoid100_trajectory_{system}.png", dpi=300)
        plt.show()
