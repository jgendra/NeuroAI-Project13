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
plt.ylabel(f'Information ({unit})', fontsize=15)
# Change xaxis label font size and remove ticks
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(0, max(v + s for v, s in zip(values, stds)) * 1.1)

# Remove top and right edges
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f'{dir}pid_bars_{SYSTEM}.png', dpi=300)
plt.show()


