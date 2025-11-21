import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

# Load data
data_dict = np.load('double_beam_data.npz')
data_all = [it[1] for it in data_dict.items()]

# LASSO coordinate descent
def lasso(XX, YY, alpha, eps, maxiter=2000):
    GG = XX @ XX.T
    HH = YY @ XX.T
    
    def softmax(x, alpha):
        if np.abs(x) > alpha:
            return x - alpha*np.sign(x)
        else:
            return 0.0
    
    WW = np.zeros_like(HH)
    WW_next = np.zeros_like(HH)
    
    for k in range(maxiter):
        for (p,q),z in np.ndenumerate(HH):
            b = HH[p,q] + WW_next[p,q]*GG[q,q] - np.dot(WW_next[p,:], GG[:,q])
            WW_next[p,q] = softmax(b, alpha) / GG[q,q]
        if la.norm(WW - WW_next) < eps:
            return WW_next
        np.copyto(WW, WW_next)
    
    return WW_next

# Create delay embeddings from 1D signal
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

# Split data into train/test
train_data = data_all[:8]
test_data = data_all[8:]

# Create delay embedded matrices
X_train_list = []
Y_train_list = []
for traj in train_data:
    X, Y = delay_embed(traj, delay)
    X_train_list.append(X)
    Y_train_list.append(Y)

X_test_list = []
Y_test_list = []
for traj in test_data:
    X, Y = delay_embed(traj, delay)
    X_test_list.append(X)
    Y_test_list.append(Y)

XX_train = np.hstack(X_train_list)
YY_train = np.hstack(Y_train_list)
XX_test = np.hstack(X_test_list)
YY_test = np.hstack(Y_test_list)

# Test different alpha values
alpha_list = [2 ** (k/2) for k in range(-20, 6)]
train_err_list = []
test_err_list = []
terms_list = []

for i, alpha in enumerate(alpha_list):
    print(f"Testing alpha {i+1}/{len(alpha_list)}: {alpha:.6f}")
    WW = lasso(XX_train, YY_train, alpha, 2**(-24))
    
    train_err = la.norm(WW @ XX_train - YY_train) / np.sqrt(XX_train.shape[1])
    test_err = la.norm(WW @ XX_test - YY_test) / np.sqrt(XX_test.shape[1])
    n_terms = np.sum(WW != 0)
    
    train_err_list.append(train_err)
    test_err_list.append(test_err)
    terms_list.append(n_terms)
    print(f"  Non-zero terms: {n_terms}, Test error: {test_err:.6f}")

# Find optimal alpha
optimal_idx = np.argmin(test_err_list)
optimal_alpha = alpha_list[optimal_idx]
optimal_n_terms = terms_list[optimal_idx]

# Plot results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.loglog(alpha_list, train_err_list, 'b-', label='Training error')
ax1.loglog(alpha_list, test_err_list, 'r-', label='Testing error')
ax1.axvline(optimal_alpha, color='g', linestyle='--', label=f'Optimal α = {optimal_alpha:.4f}')
ax1.set_xlabel('α')
ax1.set_ylabel('Error')
ax1.legend()
ax1.grid(True)

ax2.semilogx(alpha_list, terms_list, 'b-')
ax2.axvline(optimal_alpha, color='g', linestyle='--')
ax2.axhline(delay**2, color='orange', linestyle='--', label=f'Full linear model: {delay**2}')
ax2.set_xlabel('α')
ax2.set_ylabel('Number of non-zero parameters')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('problem4_sparse_regression.pdf')
plt.show()

# Report results
print(f"\nOptimal regularization parameter: α = {optimal_alpha:.6f}")
print(f"Number of non-zero parameters: {optimal_n_terms}")
print(f"Full linear model requires: {delay**2} parameters")
print(f"Reduction: {100 * (1 - optimal_n_terms / delay**2):.1f}%")
