import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import random

# Load data
data_dict = np.load('double_beam_data.npz')
data_all = [it[1] for it in data_dict.items()]

# Create delay embeddings
def delay_embed(data, delay):
    data = data.flatten()
    n_samples = len(data) - delay
    embedded = np.zeros((delay, n_samples))
    for i in range(delay):
        embedded[i, :] = data[i:i+n_samples]
    X = embedded[:, :-1]
    Y = embedded[:, 1:]
    return X, Y

delay = 12
dt = 1/5120  # sampling period

# Prepare all data
X_list = []
Y_list = []
for traj in data_all:
    X, Y = delay_embed(traj, delay)
    X_list.append(X)
    Y_list.append(Y)

XX = np.hstack(X_list)
YY = np.hstack(Y_list)

# Bagging
n_models = 100
eigenvalues_list = []

for it in range(n_models):
    # Random sample 80% of data
    id = sorted(random.sample(range(XX.shape[1]), round(0.8*XX.shape[1])))
    XX_sel = XX[:, id]
    YY_sel = YY[:, id]
    
    # Fit linear model
    KK = (YY_sel @ XX_sel.T) @ la.pinv(XX_sel @ XX_sel.T, rcond=1e-6)
    
    # Get eigenvalues
    eigvals = la.eigvals(KK)
    eigenvalues_list.append(eigvals)

eigenvalues_array = np.array(eigenvalues_list)

# For each bootstrap model, calculate frequencies and damping
omega_all = []
zeta_all = []

for eigvals in eigenvalues_array:
    # Sort by magnitude
    idx = np.argsort(np.abs(eigvals))[::-1]
    eigvals_sorted = eigvals[idx]
    
    # Calculate natural frequencies and damping ratios
    omega = np.abs(np.angle(eigvals_sorted)) / dt
    
    # Calculate damping ratio, handling division carefully
    zeta = np.zeros(len(eigvals_sorted))
    for i, lam in enumerate(eigvals_sorted):
        arg_lam = np.angle(lam)
        if np.abs(arg_lam) > 1e-10:
            zeta[i] = -np.log(np.abs(lam)) / np.abs(arg_lam)
        else:
            zeta[i] = np.nan
    
    omega_all.append(omega)
    zeta_all.append(zeta)

omega_array = np.array(omega_all)
zeta_array = np.array(zeta_all)

# Extract the two dominant modes (highest frequencies with complex conjugate pairs)
# Look at modes with non-zero imaginary part
mode1_omega = []
mode1_zeta = []
mode2_omega = []
mode2_zeta = []

for i in range(n_models):
    eigvals = eigenvalues_array[i]
    
    # Find complex conjugate pairs with largest magnitudes
    complex_mask = np.abs(np.imag(eigvals)) > 1e-10
    complex_eigvals = eigvals[complex_mask]
    
    if len(complex_eigvals) >= 2:
        # Sort by magnitude
        idx = np.argsort(np.abs(complex_eigvals))[::-1]
        complex_eigvals_sorted = complex_eigvals[idx]
        
        # Take first two unique pairs
        lam1 = complex_eigvals_sorted[0]
        arg1 = np.angle(lam1)
        omega1 = np.abs(arg1) / dt
        zeta1 = -np.log(np.abs(lam1)) / np.abs(arg1) if np.abs(arg1) > 1e-10 else np.nan
        
        # Find second mode (different frequency)
        for lam2 in complex_eigvals_sorted[1:]:
            arg2 = np.angle(lam2)
            omega2 = np.abs(arg2) / dt
            if np.abs(omega2 - omega1) > 10:  # Different mode
                zeta2 = -np.log(np.abs(lam2)) / np.abs(arg2) if np.abs(arg2) > 1e-10 else np.nan
                break
        else:
            omega2 = np.nan
            zeta2 = np.nan
        
        mode1_omega.append(omega1)
        mode1_zeta.append(zeta1)
        mode2_omega.append(omega2)
        mode2_zeta.append(zeta2)

mode1_omega = np.array(mode1_omega)
mode1_zeta = np.array(mode1_zeta)
mode2_omega = np.array(mode2_omega)
mode2_zeta = np.array(mode2_zeta)

# Remove NaN values
valid1 = np.isfinite(mode1_omega) & np.isfinite(mode1_zeta)
mode1_omega = mode1_omega[valid1]
mode1_zeta = mode1_zeta[valid1]

valid2 = np.isfinite(mode2_omega) & np.isfinite(mode2_zeta)
mode2_omega = mode2_omega[valid2]
mode2_zeta = mode2_zeta[valid2]

# Fit 2D Gaussian for each mode
print("Mode 1:")
mean1_omega = np.mean(mode1_omega)
mean1_zeta = np.mean(mode1_zeta)
data1 = np.column_stack([mode1_omega, mode1_zeta])
cov1 = np.cov(data1.T)
print(f"Mean natural frequency: {mean1_omega:.2f} rad/s")
print(f"Mean damping ratio: {mean1_zeta:.4f}")
print(f"Covariance matrix:\n{cov1}\n")

print("Mode 2:")
mean2_omega = np.mean(mode2_omega)
mean2_zeta = np.mean(mode2_zeta)
data2 = np.column_stack([mode2_omega, mode2_zeta])
cov2 = np.cov(data2.T)
print(f"Mean natural frequency: {mean2_omega:.2f} rad/s")
print(f"Mean damping ratio: {mean2_zeta:.4f}")
print(f"Covariance matrix:\n{cov2}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Mode 1
axes[0].scatter(mode1_omega, mode1_zeta, alpha=0.5, s=20, label='Bootstrap samples')
mean1 = [mean1_omega, mean1_zeta]
x = np.linspace(mean1[0]-3*np.sqrt(cov1[0,0]), mean1[0]+3*np.sqrt(cov1[0,0]), 100)
y = np.linspace(mean1[1]-3*np.sqrt(cov1[1,1]), mean1[1]+3*np.sqrt(cov1[1,1]), 100)
X_grid, Y_grid = np.meshgrid(x, y)
pos = np.dstack((X_grid, Y_grid))
rv1 = multivariate_normal(mean1, cov1)
axes[0].contour(X_grid, Y_grid, rv1.pdf(pos), levels=3, colors='red')
axes[0].plot(mean1[0], mean1[1], 'r+', markersize=15, markeredgewidth=2, label='Mean')
axes[0].set_xlabel('ω (rad/s)')
axes[0].set_ylabel('ζ')
axes[0].set_title('Mode 1')
axes[0].legend()
axes[0].grid(True)

# Mode 2
axes[1].scatter(mode2_omega, mode2_zeta, alpha=0.5, s=20, label='Bootstrap samples')
mean2 = [mean2_omega, mean2_zeta]
x = np.linspace(mean2[0]-3*np.sqrt(cov2[0,0]), mean2[0]+3*np.sqrt(cov2[0,0]), 100)
y = np.linspace(mean2[1]-3*np.sqrt(cov2[1,1]), mean2[1]+3*np.sqrt(cov2[1,1]), 100)
X_grid, Y_grid = np.meshgrid(x, y)
pos = np.dstack((X_grid, Y_grid))
rv2 = multivariate_normal(mean2, cov2)
axes[1].contour(X_grid, Y_grid, rv2.pdf(pos), levels=3, colors='red')
axes[1].plot(mean2[0], mean2[1], 'r+', markersize=15, markeredgewidth=2, label='Mean')
axes[1].set_xlabel('ω (rad/s)')
axes[1].set_ylabel('ζ')
axes[1].set_title('Mode 2')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('problem5_bagging.pdf')
plt.show()
