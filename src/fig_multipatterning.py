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

# Pattern 1
I1_2D = generate_LS_2D(pitch, 0.5, cutoff_freq, x_range)

# Pattern 2
x_range_shifted = x_range - pitch/6
I2_2D = generate_LS_2D(pitch, 0.5, cutoff_freq, x_range_shifted)

# Create wafer images by thresholding
W1 = (I1_2D > th).astype(float)
W2 = (I2_2D > th).astype(float)

# Combined pattern: W1 × (1-W2)
W_combined = W1 * (1 - W2)

# Extent for plotting
extent = [0, 2*pitch, 0, 2*pitch]

# Create figure with 3 rows, 2 columns
fig, axes = plt.subplots(3, 2, figsize=(10, 12))

# Row 1: Pattern 1
im1 = axes[0, 0].imshow(I1_2D, origin='lower', extent=extent, cmap='jet')
axes[0, 0].set_ylabel('$y$ [nm]')
axes[0, 0].set_title('Pattern 1 - Resist Image $I_1(\mathbf{x})$')
plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)

axes[0, 1].imshow(W1, origin='lower', extent=extent, cmap='binary')
axes[0, 1].set_ylabel('$y$ [nm]')
axes[0, 1].set_title('Pattern 1 - Wafer Image $W_1(\mathbf{x})$')

# Row 2: Pattern 2
im2 = axes[1, 0].imshow(I2_2D, origin='lower', extent=extent, cmap='jet')
axes[1, 0].set_ylabel('$y$ [nm]')
axes[1, 0].set_title('Pattern 2 - Resist Image $I_2(\mathbf{x})$')
plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)

axes[1, 1].imshow(W2, origin='lower', extent=extent, cmap='binary')
axes[1, 1].set_ylabel('$y$ [nm]')
axes[1, 1].set_title('Pattern 2 - Wafer Image $W_2(\mathbf{x})$')

# Row 3: Combined
# Overlay with transparency
overlay = np.zeros((*W1.shape, 3))
overlay[:, :, 0] = W2  # Red channel for W2
overlay[:, :, 2] = W1  # Blue channel for W1
axes[2, 0].imshow(overlay, origin='lower', extent=extent, alpha=0.6)
axes[2, 0].set_xlabel('$x$ [nm]')
axes[2, 0].set_ylabel('$y$ [nm]')
axes[2, 0].set_title('Overlay (both patterns)')

axes[2, 1].imshow(W_combined, origin='lower', extent=extent, cmap='binary')
axes[2, 1].set_xlabel('$x$ [nm]')
axes[2, 1].set_ylabel('$y$ [nm]')
axes[2, 1].set_title(r'Combined: $W = W_1 \times (1-W_2)$')

plt.tight_layout()
plt.savefig('fig_multipatterning_process.png', dpi=300, bbox_inches='tight')
plt.show()