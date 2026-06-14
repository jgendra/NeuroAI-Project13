import matplotlib.pyplot as plt

# Load PID results


# Plot PID bars with error bars
labels = ['Redundancy', 'Unique', 'Synergy']
values = [out['redundancy'],
        (out['unique1'] + out['unique2']) / 2,
        out['synergy']]
errors = [out['redundancy_std'],
        (out['unique1_std'] + out['unique2_std']) / 2,
        out['synergy_std']]
total  = out['mi_joint']

colors = ['#4C78A8', '#F58518', '#E45756']

fig, ax = plt.subplots(figsize=(6, 5))

bars = ax.bar(
    labels, values,
    yerr=errors,
    capsize=5,
    color=colors,
    width=0.5,
    error_kw=dict(ecolor='#333333', lw=1.2, capthick=1.2),
    zorder=3,
)

# value label just above each error bar cap
for bar, val, err in zip(bars, values, errors):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        val + err + 0.04,
        f'{val:.3f}',
        ha='center', va='bottom',
        fontsize=10, fontweight='500', color='#222222',
    )

# total info clearly above everything, anchored to axes fraction
ax.text(
    0.5, 1.06,
    f'Total  I(X_{1},X_{2} ; Y) = {total:.3f} {unit}',
    ha='center', va='bottom',
    transform=ax.transAxes,
    fontsize=13, fontweight='bold', color='#111111',
)

ax.set_ylabel(f'Information ({unit})', fontsize=11)
ax.set_title(f'PID at last timestep - {SYSTEM} RNN',
            fontsize=12, pad=28)        # extra pad keeps title below the total label
ax.tick_params(axis='x', labelsize=11, length=0)
ax.tick_params(axis='y', labelsize=10)
ax.set_ylim(0, max(v + e for v, e in zip(values, errors)) * 1.45)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.8)
ax.spines['bottom'].set_linewidth(0.8)
ax.grid(False)

fig.tight_layout()
fig.savefig(f'{dir}pid_bars_{SYSTEM}.png', bbox_inches='tight', dpi=300)
plt.show()