import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 16

# Create coordinate grid
ngridx = 200
ngridy = 200
xi = np.linspace(-3, 3, ngridx)
yi = np.linspace(-3, 3, ngridy)
Xi, Yi = np.meshgrid(xi, yi)

# Create two non-circular topographic patterns
# Pattern 1: Peak centered on the LEFT
I1 = (np.exp(-((Xi + 1.2)**2 / 1.5 + Yi**2 / 2.5)) + 
      0.3 * np.exp(-((Xi - 0.2)**2 + (Yi - 1.2)**2) / 0.6) +
      0.15 * np.sin(2 * Xi) * np.cos(Yi))

# Pattern 2: Peak centered on the RIGHT
angle = np.pi / 4
Xi_rot = Xi * np.cos(angle) - Yi * np.sin(angle)
Yi_rot = Xi * np.sin(angle) + Yi * np.cos(angle)
I2 = (np.exp(-((Xi - 1.2)**2 / 2.5 + Yi**2 / 1.5)) + 
      0.35 * np.exp(-((Xi + 0.3)**2 + (Yi + 1)**2) / 0.7) +
      0.12 * np.cos(1.5 * Xi_rot) * np.sin(1.5 * Yi_rot))

# Normalize to ensure good range for dark colors
I1 = (I1 - I1.min()) / (I1.max() - I1.min())
I2 = (I2 - I2.min()) / (I2.max() - I2.min())

# Set threshold
th = 0.5

# Shift by threshold - Images will intersect when values are 0
I1_minus_th = I1 - th
I2_minus_th = I2 - th

# Compute gradients
grad_I1_y, grad_I1_x = np.gradient(I1)
grad_I2_y, grad_I2_x = np.gradient(I2)

# Find tangency point where gradients are parallel
crossing_mask = np.abs(I1_minus_th - I2_minus_th) < 0.03
cross_product = grad_I1_x * grad_I2_y - grad_I1_y * grad_I2_x
parallel_mask = np.abs(cross_product) < 0.01
tangency_mask = crossing_mask & parallel_mask

tangency_indices = np.where(tangency_mask)
if len(tangency_indices[0]) > 0:
    center_idx = len(tangency_indices[0]) // 2
    tangency_x = xi[tangency_indices[1][center_idx]]
    tangency_y = yi[tangency_indices[0][center_idx]]
else:
    tangency_x, tangency_y = 0.0, 0.0

# Create figure
fig, ax = plt.subplots(figsize=(10, 8))

levels = 14

# Blue contour lines for I1
ax.contour(xi, yi, I1_minus_th, levels=levels, linewidths=0.5, colors='blue', alpha=0.6)

# Red contour lines for I2
ax.contour(xi, yi, I2_minus_th, levels=levels, linewidths=0.5, colors='red', alpha=0.6)

# Blue filled contours (I1) - use vmin/vmax to ensure dark blue appears
cntr1 = ax.contourf(xi, yi, I1_minus_th, levels=levels, cmap="Blues", alpha=0.7, 
                    vmin=I1_minus_th.min(), vmax=I1_minus_th.max())

# Red filled contours (I2) - use vmin/vmax to ensure dark red appears
cntr2 = ax.contourf(xi, yi, I2_minus_th, levels=levels, cmap="Reds", alpha=0.7,
                    vmin=I2_minus_th.min(), vmax=I2_minus_th.max())

# Colorbar with RdBu_r to show the concept
cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='RdBu_r', 
                    norm=plt.Normalize(vmin=I1_minus_th.min(), vmax=I1_minus_th.max())), 
                    ax=ax)
cbar.set_label('$I - th$', fontsize=16)

# Plot tangency point as fat purple dot
ax.plot(tangency_x, tangency_y, 'o', ms=15, 
        markerfacecolor='purple', markeredgecolor='black', markeredgewidth=2,
        label='Tangency: $\\nabla I_1 \\parallel \\nabla I_2$')

# Add text annotation explaining where images intersect
ax.text(0.05, 0.95, 'Images intersect when $I - th = 0$\n(zero-level contours)', 
        transform=ax.transAxes, fontsize=12, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.set(xlim=(-3, 3), ylim=(-3, 3))
ax.axis('off')  # Turn off axes
ax.set_aspect('equal')
ax.legend(loc='lower right', fontsize=12)

plt.tight_layout()
plt.savefig('fig_multipatterning_tangency.png', dpi=300, bbox_inches='tight')
plt.show()