import numpy as np
import matplotlib.pyplot as plt
from y_litho import generate_LS_1D
from y_litho import generate_LS_2D

plt.rcParams['font.size'] = 20

# Parameters
pitch = 40
cutoff_freq = 1/13.5
th = 0.5

# Generate patterns
x = np.linspace(0, 2*pitch, 2000)
I1 = generate_LS_1D(pitch, 0.5, cutoff_freq, x)
I2 = generate_LS_1D(pitch, 0.5, cutoff_freq, x - pitch/6)

# Compute gradients
dx = x[1] - x[0]
grad_I1 = np.gradient(I1, dx)
grad_I2 = np.gradient(I2, dx)

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

# Panel (a): Resist images
ax1 = axes[0]
ax1.plot(x, I1, 'b-', linewidth=2.5, label='$I_1(\mathbf{x})$')
ax1.plot(x, I2, 'r-', linewidth=2.5, label='$I_2(\mathbf{x})$')
ax1.axhline(y=th, color='k', linestyle='--', linewidth=2, label='$th$')

ax1.set_xlabel('$x$ [nm]')
ax1.set_ylabel('Resist image intensity')
ax1.set_title(r'(a) Condition 1: $I_1 - th = I_2 - th$')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 1])

# Panel (b): Gradients
ax2 = axes[1]
ax2.plot(x, grad_I1, 'b-', linewidth=2.5, label=r'$\nabla I_1$')
ax2.plot(x, grad_I2, 'r-', linewidth=2.5, label=r'$\nabla I_2$')
ax2.axhline(y=0, color='k', linestyle='--', linewidth=1.5)

ax2.set_xlabel('$x$ [nm]')
ax2.set_ylabel('Gradient')
ax2.set_title(r'(b) Condition 2: $\nabla I_1 \parallel \nabla I_2$')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fig_multipatterning_tangency.png', dpi=300, bbox_inches='tight')
plt.show()


#Figure for multi-patterning
# Generate 2D patterns
N = 300
x_range = np.linspace(0, 2*pitch, N)
y_range = np.linspace(0, 2*pitch, N)

# Pattern 1 - vertical stripes (unchanged)
I1_2D = generate_LS_2D(pitch, 0.5, cutoff_freq, x_range)

# Pattern 2 - ONE HORIZONTAL STRIPE (same thickness as vertical stripes)
# Generate using same pitch, but offset to show only one period in view
y_range_shifted = y_range + pitch/2 - pitch/6  # Offset to center one stripe
I2_1D = generate_LS_1D(pitch, 0.5, cutoff_freq, y_range_shifted)
# Tile the pattern horizontally (repeat for each column)
I2_2D = np.tile(I2_1D[:, np.newaxis], (1, N))

# Create wafer images by thresholding
W1 = (I1_2D > th).astype(float)
W2 = (I2_2D > th).astype(float)

# Combined pattern: W1 × (1-W2)
W_combined = W1 * (1 - W2)

# Extent for plotting
extent = [0, 2*pitch, 0, 2*pitch]

# Create figure with 2 rows, 3 columns + space for colorbar
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.3, hspace=0.3)

# Row 1, Col 1: Pattern 1 Resist Image
ax1 = fig.add_subplot(gs[0, 0])
im1 = ax1.imshow(I1_2D, origin='lower', extent=extent, cmap='jet', vmin=0, vmax=1)
ax1.set_ylabel('$y$ [nm]')
ax1.set_title('(a)', loc='left', fontweight='bold')
ax1.tick_params(labelbottom=False)
ax1.text(0.05, 0.95, '$I_1(\mathbf{x})$', transform=ax1.transAxes,
                fontsize=18, verticalalignment='top',
                bbox=dict(boxstyle='square', facecolor='white', alpha=1.0, edgecolor='black'))

# Row 1, Col 2: Pattern 2 Resist Image
ax2 = fig.add_subplot(gs[0, 1])
im2 = ax2.imshow(I2_2D, origin='lower', extent=extent, cmap='jet', vmin=0, vmax=1)
ax2.set_title('(b)', loc='left', fontweight='bold')
ax2.tick_params(labelbottom=False, labelleft=False)
ax2.set_yticks([])
ax2.text(0.05, 0.95, '$I_2(\mathbf{x})$', transform=ax2.transAxes,
                fontsize=18, verticalalignment='top',
                bbox=dict(boxstyle='square', facecolor='white', alpha=1.0, edgecolor='black'))

# Row 1, Col 3: Overlay of both patterns
ax3 = fig.add_subplot(gs[0, 2])
overlay = np.zeros((*W1.shape, 3))
overlay[:, :, 0] = W2  # Red channel for W2
overlay[:, :, 2] = W1  # Blue channel for W1
ax3.imshow(overlay, origin='lower', extent=extent, alpha=0.6)
ax3.set_title('(c)', loc='left', fontweight='bold')
ax3.tick_params(labelbottom=False, labelleft=False)
ax3.set_yticks([])
ax3.text(0.05, 0.95, 'Overlay', transform=ax3.transAxes,
                fontsize=18, verticalalignment='top',
                bbox=dict(boxstyle='square', facecolor='white', alpha=1.0, edgecolor='black'))

# Colorbar for first row only
cbar_ax = fig.add_subplot(gs[0, 3])
fig.colorbar(im1, cax=cbar_ax)

# Row 2, Col 1: Pattern 1 Wafer Image
ax4 = fig.add_subplot(gs[1, 0])
ax4.imshow(W1, origin='lower', extent=extent, cmap='binary')
ax4.set_xlabel('$x$ [nm]')
ax4.set_ylabel('$y$ [nm]')
ax4.set_title('(d)', loc='left', fontweight='bold')
ax4.text(0.05, 0.95, '$W_1(\mathbf{x})$', transform=ax4.transAxes,
                fontsize=18, verticalalignment='top',
                bbox=dict(boxstyle='square', facecolor='white', alpha=1.0, edgecolor='black'))

# Row 2, Col 2: Pattern 2 Wafer Image
ax5 = fig.add_subplot(gs[1, 1])
ax5.imshow(W2, origin='lower', extent=extent, cmap='binary')
ax5.set_xlabel('$x$ [nm]')
ax5.set_title('(e)', loc='left', fontweight='bold')
ax5.tick_params(labelleft=False)
ax5.set_yticks([])
ax5.text(0.05, 0.95, '$W_2(\mathbf{x})$', transform=ax5.transAxes,
                fontsize=18, verticalalignment='top',
                bbox=dict(boxstyle='square', facecolor='white', alpha=1.0, edgecolor='black'))

# Row 2, Col 3: Combined Pattern
ax6 = fig.add_subplot(gs[1, 2])
ax6.imshow(W_combined, origin='lower', extent=extent, cmap='binary')
ax6.set_xlabel('$x$ [nm]')
ax6.set_title('(f)', loc='left', fontweight='bold')
ax6.tick_params(labelleft=False)
ax6.set_yticks([])
ax6.text(0.05, 0.95, '$W = W_1 \\times (1-W_2)$', transform=ax6.transAxes,
                fontsize=18, verticalalignment='top',
                bbox=dict(boxstyle='square', facecolor='white', alpha=1.0, edgecolor='black'))

plt.savefig('fig_multipatterning_process.png', dpi=300, bbox_inches='tight')
plt.show()