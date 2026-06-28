import matplotlib.pyplot as plt
import numpy as np

# Define input data directory
dir = "src/analysis/results/"

# Load PID results
SYSTEM = "Sinusoid"
pid_path = f"{dir}pid_{SYSTEM}.npz"
out = np.load(pid_path)

# Plot PID bars with std error bars, and total MI as a title annotation
unit = 'bits'

labels = ['Redundancy', 'Unique1', 'Unique2', 'Synergy']
values = [out['redundancy'], out['unique1'], out['unique2'], out['synergy']]
stds = [out['redundancy_std'], out['unique1_std'], out['unique2_std'], out['synergy_std']]
total = out['mi_joint']

plt.figure(figsize=(8, 6))
bars = plt.bar(labels, values, yerr=stds, capsize=5, width=0.6)
# Add value labels above bars
for bar, val, std in zip(bars, values, stds):
    plt.text(bar.get_x() + bar.get_width() / 2, val + std + 0.02, f'{val:.3f}', ha='center', va='bottom', fontsize=12)
# Give different colors to each bar
colors = ['#4C78A8', '#F58518', '#E45756', '#54A24B']
for bar, color in zip(bars, colors):
    bar.set_color(color)

plt.title(f'PID at last timestep - {SYSTEM} (10 unit) RNN\nTotal I(X1,X2;Y) = {total:.3f} {unit}', fontsize=15, fontweight='bold')
# Fontsize and limits
plt.ylabel(f'Information ({unit})', fontsize=15)
plt.yticks(fontsize=12)
plt.ylim(0, max(v + s for v, s in zip(values, stds)) * 1.1)
plt.xticks(fontsize=15)

# Remove top and right edges
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
# plt.savefig(f'{dir}pid_bars_{SYSTEM}.png', dpi=300)
plt.show()




# Measure Gaussianity of hidden unit activations across all timesteps, for each unit.
import torch
from scipy import stats

h = torch.load(f'{dir}hidden_Sinusoid.pt', weights_only=True).numpy()  # (300, 10)
n_units = h.shape[1]

fig, axes = plt.subplots(2, 5, figsize=(14, 5))
axes = axes.ravel()

for i in range(n_units):
    ax = axes[i]
    unit_acts = h[:, i]                          # 300 values across timesteps

    # histogram of activations
    ax.hist(unit_acts, bins=20, color='#4C78A8', alpha=0.75, density=True, zorder=2)

    # overlay best-fit Gaussian for visual comparison
    mu, sigma = unit_acts.mean(), unit_acts.std()
    x = np.linspace(unit_acts.min(), unit_acts.max(), 200)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), color='#E45756', lw=1.8, zorder=3)

    # Shapiro-Wilk p-value (tests departure from normality; p<0.05 = non-Gaussian)
    _, p = stats.shapiro(unit_acts)
    ax.set_title(f'Unit {i+1}   p={p:.2f}', fontsize=9)
    ax.tick_params(labelsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle('Hidden unit activation distributions (blue=data, red=fitted Gaussian)\n'
             'Shapiro-Wilk p-value: p>0.05 consistent with Gaussian', fontsize=11)
fig.tight_layout()
# fig.savefig('unit_activations_gaussianity.png', dpi=150, bbox_inches='tight')
plt.show()