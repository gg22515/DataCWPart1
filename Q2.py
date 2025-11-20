import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import random

random.seed(100)
train_test_split = 0.8

# Loading data
data_dict = np.load("double_beam_data.npz")
data_all = [it[1] for it in data_dict.items()]

random.shuffle(data_all)
split = int(len(data_all) * train_test_split)
training_data = data_all[:split]
testing_data = data_all[split:]

def delay_embed(trajectory, delay):
    XX = np.zeros((trajectory.shape[0] * delay, trajectory.shape[1] - delay + 1))
    for k in range(delay):
        XX[k * trajectory.shape[0]:(k + 1) * trajectory.shape[0], :] = \
            trajectory[:, k:k + trajectory.shape[1] - delay + 1]
    return XX

def delay_embed_all(trajectories, delay):
    XX_list = [delay_embed(traj[:, :-1], delay) for traj in trajectories]
    YY_list = [delay_embed(traj[:, 1:], delay) for traj in trajectories]
    XX = np.hstack(XX_list)
    YY = np.hstack(YY_list)
    return XX, YY

def DMDModel(data, delay_length):
    XX, YY = delay_embed_all(data, delay_length)
    K = (YY @ XX.T) @ la.pinv(XX @ XX.T, rcond=1e-6)
    return K, XX, YY

# Find max DMD rank
K, XX, YY = DMDModel(training_data, 32)

values, vectors = la.eig(K.T)
right_vectors = la.inv(vectors)

GG = XX @ XX.T
HH = YY @ XX.T
LL = YY @ YY.T

res_num = [vectors[:,k].conj().T @ (LL - values[k] * HH - np.conj(values[k]) * HH.T + (np.abs(values[k]) ** 2) * GG) @ vectors[:,k] for k in range(vectors.shape[0])]
res_den = [vectors[:,k].conj().transpose() @ GG @ vectors[:,k] for k in range(vectors.shape[0])]
res = np.real(np.array(res_num) / np.array(res_den))

sorted_residuals = np.sort(res)
residual_threshold = 1e-1

max_dmd_rank = len(sorted_residuals)
for i, val in enumerate(sorted_residuals):
    if val > residual_threshold:
        print("Max DMD Rank:", i)
        max_dmd_rank = i
        break

plt.figure(figsize=(10, 6))
x_positions = range(1, len(sorted_residuals) + 1)
plt.bar(x_positions, sorted_residuals)
plt.title("DMD Residuals (sorted)")
plt.xlabel("Mode Index")
plt.ylabel("Residual")
plt.xticks(x_positions)
plt.savefig("DMDResidBar.png")
plt.show()

# Grid search functions

def fit_dmd(training_data, delay, dmd_rank):
    XX, YY = delay_embed_all(training_data, delay)
    K = (YY @ XX.T) @ la.pinv(XX @ XX.T, rcond=1e-6)
    
    values, vectors = la.eig(K.T)
    right_vectors = la.inv(vectors)
    
    GG = XX @ XX.T
    HH = YY @ XX.T
    LL = YY @ YY.T
    
    res_num = [vectors[:,k].conj().T @ (LL - values[k] * HH - np.conj(values[k]) * HH.T + (np.abs(values[k]) ** 2) * GG) @ vectors[:,k] for k in range(vectors.shape[0])]
    res_den = [vectors[:,k].conj().transpose() @ GG @ vectors[:,k] for k in range(vectors.shape[0])]
    res = np.real(np.array(res_num) / np.array(res_den))
    
    id = np.argsort(res)
    DMD_proj = right_vectors[id[0:dmd_rank],:].T @ vectors[:,id[0:dmd_rank]].T
    K_proj = np.real(K @ DMD_proj)
    
    return K_proj

def test_method_1(K_proj, testing_data, delay):
    XX_test, YY_test = delay_embed_all(testing_data, delay)
    YY_pred = K_proj @ XX_test
    return np.mean((YY_pred[-1, :] - YY_test[-1, :]) ** 2)

def iterate_dmd_trajectory(initial_condition, K_proj, n_steps):
    current_state = initial_condition.reshape(-1, 1)
    trajectory = [current_state[0, 0]]
    
    for step in range(n_steps - 1):
        next_state = K_proj @ current_state
        trajectory.append(next_state[0, 0])
        current_state = next_state
    
    return np.array(trajectory)

def test_method_2(K_proj, testing_data, delay):
    errors = []
    for test_traj in testing_data:
        test_signal = test_traj[0, :]
        initial_condition = test_signal[:delay]
        n_predict = min(5000, len(test_signal) - delay)
        
        if n_predict <= 0:
            continue
            
        predicted = iterate_dmd_trajectory(initial_condition, K_proj, n_predict)
        actual = test_signal[:n_predict]
        
        error = np.mean((predicted - actual)**2)
        errors.append(error)
    
    return np.mean(errors) if errors else np.inf

def get_top_5(error_matrix, delay_range, rank_range):
    valid_errors = []
    
    for d_idx, delay in enumerate(delay_range):
        for r_idx, rank in enumerate(rank_range):
            error = error_matrix[d_idx, r_idx]
            if not np.isnan(error) and not np.isinf(error):
                valid_errors.append((delay, rank, error, d_idx, r_idx))
    
    valid_errors.sort(key=lambda x: x[2])
    return valid_errors[:5]

# Grid search

print("Starting grid search for DMD testing method comparison")
max_delay = 32
delay_range = range(1, max_delay + 1)
rank_range = range(1, max_dmd_rank)

errors_method1 = np.zeros((len(delay_range), len(rank_range)))
errors_method2 = np.zeros((len(delay_range), len(rank_range)))

for d_idx, delay in enumerate(delay_range):
    if delay % 5 == 0:
        print(delay,"/",max_delay)
    
    for r_idx, rank in enumerate(rank_range):
        try:
            K_proj = fit_dmd(training_data, delay, rank)
            errors_method1[d_idx, r_idx] = test_method_1(K_proj, testing_data, delay)
            errors_method2[d_idx, r_idx] = test_method_2(K_proj, testing_data, delay)
        except:
            errors_method1[d_idx, r_idx] = np.nan
            errors_method2[d_idx, r_idx] = np.nan

top_5_method1 = get_top_5(errors_method1, delay_range, rank_range)
top_5_method2 = get_top_5(errors_method2, delay_range, rank_range)

# Methods comparison heatmap

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

im1 = axes[0].imshow(
    np.log10(errors_method1 + 1e-12), 
    aspect="auto", origin="lower", cmap="viridis",
    extent=[min(rank_range)-0.5, max(rank_range)+0.5, 
            min(delay_range)-0.5, max(delay_range)+0.5]
)
axes[0].set_xlabel("DMD Rank")
axes[0].set_ylabel("Delay length")
axes[0].set_title("1 Step prediction")
axes[0].set_xticks(list(rank_range))
axes[0].set_yticks(list(delay_range))
axes[0].set_xticks([x-0.5 for x in rank_range] + [max(rank_range)+0.5], minor=True)
axes[0].set_yticks([y-0.5 for y in delay_range] + [max(delay_range)+0.5], minor=True)
axes[0].grid(which="minor", alpha=0.8, linewidth=0.5)
cbar1 = plt.colorbar(im1, ax=axes[0])
cbar1.set_label("error", rotation=90)

for i, (delay, rank, error, row, col) in enumerate(top_5_method1):
    axes[0].text(rank, delay, str(i+1), color="white", ha="center", va="center", fontweight="bold")

im2 = axes[1].imshow(
    np.log10(errors_method2 + 1e-12), 
    aspect="auto", origin="lower", cmap="viridis",
    extent=[min(rank_range)-0.5, max(rank_range)+0.5, 
            min(delay_range)-0.5, max(delay_range)+0.5]
)
axes[1].set_xlabel("DMD Rank")
axes[1].set_ylabel("Delay length")
axes[1].set_title("Long term prediction")
axes[1].set_xticks(list(rank_range))
axes[1].set_yticks(list(delay_range))
axes[1].set_xticks([x-0.5 for x in rank_range] + [max(rank_range)+0.5], minor=True)
axes[1].set_yticks([y-0.5 for y in delay_range] + [max(delay_range)+0.5], minor=True)
axes[1].grid(which="minor", alpha=0.8, linewidth=0.5)
cbar2 = plt.colorbar(im2, ax=axes[1])
cbar2.set_label("error", rotation=90)

for i, (delay, rank, error, row, col) in enumerate(top_5_method2):
    axes[1].text(rank, delay, str(i+1), color="white", ha="center", va="center", fontweight="bold")

plt.tight_layout()
plt.savefig("DMDMethodComp.png")
plt.show()

print("Top 5 for Method 1 (1 step prediction):")
for i, (delay, rank, error, _, _) in enumerate(top_5_method1):
    print(i + 1, "Delay=", delay, "Rank=",rank, "MSE=",error)

print("Top 5 for Method 2 (Long term prediction):")
for i, (delay, rank, error, _, _) in enumerate(top_5_method2):
    print(i + 1, "Delay=", delay, "Rank=",rank, "MSE=",error)


# N-step ahead prediction method
steps_ahead = 10  # Change this variable to predict different number of steps ahead

def delay_embed_all_nstep(trajectories, delay, steps_ahead):
    XX_list = [delay_embed(traj[:, :-steps_ahead], delay) for traj in trajectories]  # input
    YY_list = [delay_embed(traj[:, steps_ahead:], delay) for traj in trajectories]   # N steps ahead
    XX = np.hstack(XX_list)
    YY = np.hstack(YY_list)
    return XX, YY

def fit_dmd_nstep(training_data, delay, dmd_rank, steps_ahead):
    XX, YY = delay_embed_all_nstep(training_data, delay, steps_ahead)
    K = (YY @ XX.T) @ la.pinv(XX @ XX.T, rcond=1e-6)
    
    values, vectors = la.eig(K.T)
    right_vectors = la.inv(vectors)
    
    GG = XX @ XX.T
    HH = YY @ XX.T
    LL = YY @ YY.T
    
    res_num = [vectors[:,k].conj().T @ (LL - values[k] * HH - np.conj(values[k]) * HH.T + (np.abs(values[k]) ** 2) * GG) @ vectors[:,k] for k in range(vectors.shape[0])]
    res_den = [vectors[:,k].conj().transpose() @ GG @ vectors[:,k] for k in range(vectors.shape[0])]
    res = np.real(np.array(res_num) / np.array(res_den))
    
    id = np.argsort(res)
    DMD_proj = right_vectors[id[0:dmd_rank],:].T @ vectors[:,id[0:dmd_rank]].T
    K_proj = np.real(K @ DMD_proj)
    
    return K_proj

def test_method_nstep(K_proj, testing_data, delay, steps_ahead):
    XX_test, YY_test = delay_embed_all_nstep(testing_data, delay, steps_ahead)
    YY_pred = K_proj @ XX_test
    return np.mean((YY_pred[-1, :] - YY_test[-1, :]) ** 2)

# Grid search for N-step method
print(f"Starting grid search for {steps_ahead}-step prediction")
errors_method_nstep = np.zeros((len(delay_range), len(rank_range)))

for d_idx, delay in enumerate(delay_range):
    if delay % 5 == 0:
        print(f"{steps_ahead}-step:", delay,"/",max_delay)
    
    for r_idx, rank in enumerate(rank_range):
        try:
            K_proj_nstep = fit_dmd_nstep(training_data, delay, rank, steps_ahead)
            errors_method_nstep[d_idx, r_idx] = test_method_nstep(K_proj_nstep, testing_data, delay, steps_ahead)
        except:
            errors_method_nstep[d_idx, r_idx] = np.nan

top_5_method_nstep = get_top_5(errors_method_nstep, delay_range, rank_range)

# Three-way comparison heatmap
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# 1-step prediction
im1 = axes[0].imshow(
    np.log10(errors_method1 + 1e-12), 
    aspect="auto", origin="lower", cmap="viridis",
    extent=[min(rank_range)-0.5, max(rank_range)+0.5, 
            min(delay_range)-0.5, max(delay_range)+0.5]
)
axes[0].set_xlabel("DMD Rank")
axes[0].set_ylabel("Delay length")
axes[0].set_title("1 Step prediction")
axes[0].set_xticks(list(rank_range))
axes[0].set_yticks(list(delay_range))
axes[0].set_xticks([x-0.5 for x in rank_range] + [max(rank_range)+0.5], minor=True)
axes[0].set_yticks([y-0.5 for y in delay_range] + [max(delay_range)+0.5], minor=True)
axes[0].grid(which="minor", alpha=0.8, linewidth=0.5)
cbar1 = plt.colorbar(im1, ax=axes[0])
cbar1.set_label("error", rotation=90)

for i, (delay, rank, error, row, col) in enumerate(top_5_method1):
    axes[0].text(rank, delay, str(i+1), color="white", ha="center", va="center", fontweight="bold")

# N-step prediction
im2 = axes[1].imshow(
    np.log10(errors_method_nstep + 1e-12), 
    aspect="auto", origin="lower", cmap="viridis",
    extent=[min(rank_range)-0.5, max(rank_range)+0.5, 
            min(delay_range)-0.5, max(delay_range)+0.5]
)
axes[1].set_xlabel("DMD Rank")
axes[1].set_ylabel("Delay length")
axes[1].set_title(f"{steps_ahead} Step prediction")
axes[1].set_xticks(list(rank_range))
axes[1].set_yticks(list(delay_range))
axes[1].set_xticks([x-0.5 for x in rank_range] + [max(rank_range)+0.5], minor=True)
axes[1].set_yticks([y-0.5 for y in delay_range] + [max(delay_range)+0.5], minor=True)
axes[1].grid(which="minor", alpha=0.8, linewidth=0.5)
cbar2 = plt.colorbar(im2, ax=axes[1])
cbar2.set_label("error", rotation=90)

for i, (delay, rank, error, row, col) in enumerate(top_5_method_nstep):
    axes[1].text(rank, delay, str(i+1), color="white", ha="center", va="center", fontweight="bold")

# Long term prediction
im3 = axes[2].imshow(
    np.log10(errors_method2 + 1e-12), 
    aspect="auto", origin="lower", cmap="viridis",
    extent=[min(rank_range)-0.5, max(rank_range)+0.5, 
            min(delay_range)-0.5, max(delay_range)+0.5]
)
axes[2].set_xlabel("DMD Rank")
axes[2].set_ylabel("Delay length")
axes[2].set_title("Long term prediction")
axes[2].set_xticks(list(rank_range))
axes[2].set_yticks(list(delay_range))
axes[2].set_xticks([x-0.5 for x in rank_range] + [max(rank_range)+0.5], minor=True)
axes[2].set_yticks([y-0.5 for y in delay_range] + [max(delay_range)+0.5], minor=True)
axes[2].grid(which="minor", alpha=0.8, linewidth=0.5)
cbar3 = plt.colorbar(im3, ax=axes[2])
cbar3.set_label("error", rotation=90)

for i, (delay, rank, error, row, col) in enumerate(top_5_method2):
    axes[2].text(rank, delay, str(i+1), color="white", ha="center", va="center", fontweight="bold")

plt.tight_layout()
plt.savefig("DMDThreeWayComp.png")
plt.show()

print(f"Top 5 for {steps_ahead}-Step prediction:")
for i, (delay, rank, error, _, _) in enumerate(top_5_method_nstep):
    print(i + 1, "Delay=", delay, "Rank=",rank, "MSE=",error)
