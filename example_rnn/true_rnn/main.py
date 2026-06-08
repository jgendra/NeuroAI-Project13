import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from example_rnn.true_rnn.trueRNN import TrueRNN

# Define task directory
dir = "example_rnn/true_rnn/"

# hyperparameters
BATCH_SIZE = 1   # not really used here, since we train on a single trajectory
EPOCHS = 1000   
LEARNING_RATE = 1e-3

# Choose dynamical system
system = 'Lorenz'  # 'VanDerPol' or 'Lorenz'

if __name__ == "__main__":

    y_true = torch.load(f'{dir}y_{system}.pt', weights_only=True)  # Size (300, 2) or (300, 3)
    # print(y_true.shape)

    n_timesteps, dim = y_true.shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_true = y_true.to(device)

    # Create RNN
    model = TrueRNN(dim=dim, system=system).to(device)

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # create the optimizer
    loss_function = nn.MSELoss()  # loss function (MSE)

     # For tracking over epochs
    training_losses = []
    jacobian_norm_max_list = []


    # ===== TRAINING LOOP =====
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()

        # Teacher forcing over the full sequence: predict x_{t+1} from x_t
        x_in = y_true[:-1]
        target = y_true[1:]
        pred, _ = model(x_in)
        loss = loss_function(pred, target)

        # Backpropagate once after summing over all timesteps
        loss.backward()
        optimizer.step()

        # ---- Track Jacobian after this epoch ----
        model.eval()
        # do NOT use torch.no_grad() here, autograd is needed inside calculate_jacobian
        with torch.enable_grad():
            # detach so we don't reuse the training graph
            x_for_jacobian = y_true.detach()
            jacobian = model.calculate_jacobian(x_for_jacobian)   # shape: (n_timesteps, dim, dim)
            # Frobenius norm over last two dims -> shape (n_timesteps,)
            jacobian_norms = torch.linalg.norm(jacobian, dim=(1, 2))
            # maximum norm over all timesteps
            jacobian_norm_max = jacobian_norms.max().item()

        # Print loss and max Jacobian norm every 50 epochs
        training_losses.append(loss.item())
        jacobian_norm_max_list.append(jacobian_norm_max)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1}/{EPOCHS}, "
                f"training loss = {loss.item():.6e}, "
                f"max Jacobian norm = {jacobian_norm_max:.6e}"
            )

    # Save trained RNN
    torch.save(model.state_dict(), f"{dir}RNN_{system}.pt")
    print(f"Trained RNN saved to {dir}RNN_{system}.pt")


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

        mse_eval = loss_function(y_pred, y_true).item()
        print(f"Evaluation MSE over full trajectory: {mse_eval:.6e}")


    # ===== PLOTTING TRUE VS PREDICTED TRAJECTORY =====
    y_true_np = y_true.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()
    time = range(n_timesteps)

    # Plot for Van der Pol system
    if system == 'VanDerPol':
        plt.figure(figsize=(10, 6))
        plt.plot(time, y_true_np[:, 0], label='y1_true', c='C0')
        plt.plot(time, y_pred_np[:, 0], '--', label='y1_predicted', c='C1')
        plt.plot(time, y_true_np[:, 1], label='y2_true', c='C0')
        plt.plot(time, y_pred_np[:, 1], '--', label='y2_predicted', c='C1')

        plt.ylabel(f'y1, y2')
        plt.title('Van der Pol trajectory: true vs predicted')
        plt.xlabel('Time (s)')
        plt.legend(loc='best')
 
        plt.tight_layout()
        plt.show()

    # Plot for Lorenz system
    elif system == 'Lorenz':
        from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(y_true_np[:, 0], y_true_np[:, 1], y_true_np[:, 2], label='True trajectory', c='C0')
        ax.plot(y_pred_np[:, 0], y_pred_np[:, 1], y_pred_np[:, 2], '--', label='Predicted trajectory', c='C1')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Lorenz trajectory: true vs predicted')
        ax.legend(loc='best')

        plt.tight_layout()
        plt.show()
    

