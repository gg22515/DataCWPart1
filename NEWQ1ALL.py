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

def PCAModel(data, delay_length, pca_rank):
    X, Y = delay_embed_all(data, delay_length)
    
    mean_x = X.mean(axis=1, keepdims=True)
    mean_y = Y.mean(axis=1, keepdims=True)
    
    B_x = X - mean_x
    C = B_x @ B_x.T
    U, S, _ = la.svd(C)
    U_red = U[:, :pca_rank]

    B_y = Y - mean_y
    T_x = U_red.T @ B_x
    T_y = U_red.T @ B_y
    A = T_y @ T_x.T @ la.pinv(T_x @ T_x.T)

    return A, mean_x, mean_y, U_red

#Find max PCA rank

A_PCA, _, _, _ = PCAModel(training_data, 32, 16)
importance = np.linalg.norm(A_PCA, axis=1) * np.linalg.norm(A_PCA, axis=0)
mean_importance = importance.mean(axis=0)
importance_threshold = 0.05

max_pca_rank = 16
for i, val in enumerate(importance):
    if val < importance_threshold * mean_importance:
        print("Max PCA Rank:", i + 1)  
        max_pca_rank = i + 1
        break

plt.figure(figsize=(10, 6))
x_positions = range(1, 17)
plt.bar(x_positions, importance)
plt.title("Influence of each PC")
plt.xlabel("PCs")
plt.ylabel("Influence")
plt.xticks(x_positions)  
plt.savefig("MaxPCAGraph.png")
plt.show()

#Delay length graph

max_delay = 32
rank_ar = []

for delay_pointer in range(1, max_delay + 1):
    A_delay, _, _, _ = PCAModel(training_data, delay_pointer, max_pca_rank)
    matrix_rank = la.matrix_rank(A_delay, tol=1e-6)
    rank_ar.append(matrix_rank)

plt.figure(figsize=(10, 6))
plt.plot(range(1, max_delay + 1), rank_ar)
plt.title("Delay length vs rank")
plt.xlabel("Delay length")
plt.ylabel("Rank")
plt.grid()
plt.savefig("MaxDelayGraph.png")
plt.show()

#Grid search functions

def fit_pca(X_train, Y_train, pca_rank):
    mean_X = X_train.mean(axis=1, keepdims=True)
    mean_Y = Y_train.mean(axis=1, keepdims=True)
    
    X_centered = X_train - mean_X
    Y_centered = Y_train - mean_Y
    
    C = X_centered @ X_centered.T
    U, S, _ = la.svd(C, hermitian=True)
    U_red = U[:, :pca_rank]
    
    X_pca = U_red.T @ X_centered
    Y_pca = U_red.T @ Y_centered
    A = (Y_pca @ X_pca.T) @ la.pinv(X_pca @ X_pca.T, rcond=1e-9)
    
    return A, U_red, mean_X, mean_Y

def test_method_1(A, U, mean_x, mean_y, testing_data, delay):
    XX_test, YY_test = delay_embed_all(testing_data, delay)
    T_x_test = U.T @ (XX_test - mean_x)
    T_y_pred = A @ T_x_test
    y_pred = U @ T_y_pred + mean_y
    return np.mean((y_pred[-1, :] - YY_test[-1, :]) ** 2)

def iterate_trajectory(initial_condition, A, mean_x, mean_y, U_red, n_steps):
    current_state = initial_condition.reshape(-1, 1)
    trajectory = [current_state[0, 0]]
    
    for step in range(n_steps - 1):
        centered_state = current_state - mean_x
        pc_state = U_red.T @ centered_state
        pc_next = A @ pc_state
        centered_next = U_red @ pc_next
        next_state = centered_next + mean_y
        trajectory.append(next_state[0, 0])
        current_state = next_state
    
    return np.array(trajectory)

def test_method_2(A, U, mean_x, mean_y, testing_data, delay):
    errors = []
    for test_traj in testing_data:
        test_signal = test_traj[0, :]
        initial_condition = test_signal[:delay]
        n_predict = min(5000, len(test_signal) - delay)
        
        if n_predict <= 0:
            continue
            
        predicted = iterate_trajectory(initial_condition, A, mean_x, mean_y, U, n_predict)
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

#Grid search

print("Starting grid search for testing method comparison")
delay_range = range(1, max_delay + 1)
rank_range = range(1, max_pca_rank)

errors_method1 = np.zeros((len(delay_range), len(rank_range)))
errors_method2 = np.zeros((len(delay_range), len(rank_range)))

for d_idx, delay in enumerate(delay_range):
    XX, YY = delay_embed_all(training_data, delay)
    if delay % 5 == 0:
        print(delay,"/",max_delay)
    
    for r_idx, rank in enumerate(rank_range):
        if rank > XX.shape[0]:
            errors_method1[d_idx, r_idx] = np.nan
            errors_method2[d_idx, r_idx] = np.nan
            continue
            
        try:
            A, U, mean_x, mean_y = fit_pca(XX, YY, rank)
            errors_method1[d_idx, r_idx] = test_method_1(A, U, mean_x, mean_y, testing_data, delay)
            errors_method2[d_idx, r_idx] = test_method_2(A, U, mean_x, mean_y, testing_data, delay)
        except:
            errors_method1[d_idx, r_idx] = np.nan
            errors_method2[d_idx, r_idx] = np.nan

top_5_method1 = get_top_5(errors_method1, delay_range, rank_range)
top_5_method2 = get_top_5(errors_method2, delay_range, rank_range)

#Methods comparison heatmap

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

im1 = axes[0].imshow(
    np.log10(errors_method1 + 1e-12), 
    aspect="auto", origin="lower", cmap="viridis",
    extent=[min(rank_range)-0.5, max(rank_range)+0.5, 
            min(delay_range)-0.5, max(delay_range)+0.5]
)
axes[0].set_xlabel("PCA Rank")
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
axes[1].set_xlabel("PCA Rank")
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
plt.savefig("MethodsComparisonHeatmap.png")
plt.show()

print("Top 5 for Method 1 (1 step prediction):")
for i, (delay, rank, error, _, _) in enumerate(top_5_method1):
    print(i + 1, "Delay=", delay, "Rank=",rank, "MSE=",error)

print("Top 5 for Method 2 (Long term prediction):")
for i, (delay, rank, error, _, _) in enumerate(top_5_method2):
    print(i + 1, "Delay=", delay, "Rank=",rank, "MSE=",error)

#Training vs testing error

def test_method_2_on_training(A, U, mean_x, mean_y, training_data, delay):
    errors = []
    for train_traj in training_data:
        test_signal = train_traj[0, :]
        initial_condition = test_signal[:delay]
        n_predict = min(1000, len(test_signal) - delay)
        
        if n_predict <= 0:
            continue
            
        predicted = iterate_trajectory(initial_condition, A, mean_x, mean_y, U, n_predict)
        actual = test_signal[:n_predict]
        
        error = np.mean((predicted - actual)**2)
        errors.append(error)
    
    return np.mean(errors) if errors else np.inf

errors_training = np.zeros((len(delay_range), len(rank_range)))

print("Starting grid search for training and testing error")

for d_idx, delay in enumerate(delay_range):
    XX, YY = delay_embed_all(training_data, delay)
    if delay % 5 == 0:
        print(delay,"/",max_delay)
    
    for r_idx, rank in enumerate(rank_range):
        if rank > XX.shape[0]:
            errors_training[d_idx, r_idx] = np.nan
            continue
            
        try:
            A, U, mean_x, mean_y = fit_pca(XX, YY, rank)
            errors_training[d_idx, r_idx] = test_method_2_on_training(A, U, mean_x, mean_y, training_data, delay)
        except:
            errors_training[d_idx, r_idx] = np.nan

top_5_training = get_top_5(errors_training, delay_range, rank_range)
top_5_testing = get_top_5(errors_method2, delay_range, rank_range)

#Training vs testing heatmap

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

im1 = axes[0].imshow(
    np.log10(errors_training + 1e-12), 
    aspect="auto", origin="lower", cmap="viridis",
    extent=[min(rank_range)-0.5, max(rank_range)+0.5, 
            min(delay_range)-0.5, max(delay_range)+0.5]
)
axes[0].set_xlabel("PCA Rank")
axes[0].set_ylabel("Delay length")
axes[0].set_title("Training error")
axes[0].set_xticks(list(rank_range))
axes[0].set_yticks(list(delay_range))
axes[0].set_xticks([x-0.5 for x in rank_range] + [max(rank_range)+0.5], minor=True)
axes[0].set_yticks([y-0.5 for y in delay_range] + [max(delay_range)+0.5], minor=True)
axes[0].grid(which="minor", alpha=0.8, linewidth=0.5)
cbar1 = plt.colorbar(im1, ax=axes[0])
cbar1.set_label("error", rotation=90)

for i, (delay, rank, error, row, col) in enumerate(top_5_training):
    axes[0].text(rank, delay, str(i+1), color="white", ha="center", va="center", fontweight="bold")

im2 = axes[1].imshow(
    np.log10(errors_method2 + 1e-12), 
    aspect="auto", origin="lower", cmap="viridis",
    extent=[min(rank_range)-0.5, max(rank_range)+0.5, 
            min(delay_range)-0.5, max(delay_range)+0.5]
)
axes[1].set_xlabel("PCA Rank")
axes[1].set_ylabel("Delay length")
axes[1].set_title("Testing error")
axes[1].set_xticks(list(rank_range))
axes[1].set_yticks(list(delay_range))
axes[1].set_xticks([x-0.5 for x in rank_range] + [max(rank_range)+0.5], minor=True)
axes[1].set_yticks([y-0.5 for y in delay_range] + [max(delay_range)+0.5], minor=True)
axes[1].grid(which="minor", alpha=0.8, linewidth=0.5)
cbar2 = plt.colorbar(im2, ax=axes[1])
cbar2.set_label("error", rotation=90)

for i, (delay, rank, error, row, col) in enumerate(top_5_testing):
    axes[1].text(rank, delay, str(i+1), color="white", ha="center", va="center", fontweight="bold")

plt.tight_layout()
plt.savefig("TrainTestHeatmap.png")
plt.show()

print("Top 5 for training error:")
for i, (delay, rank, error, _, _) in enumerate(top_5_training):
    print(i + 1, "Delay=", delay, "Rank=",rank, "MSE=",error)

print("Top 5 for testing error:")
for i, (delay, rank, error, _, _) in enumerate(top_5_testing):
    print(i + 1, "Delay=", delay, "Rank=",rank, "MSE=",error)

#Predicted vs actual signal

best_delay, best_rank, _, _, _ = top_5_method2[0]

XX, YY = delay_embed_all(training_data, best_delay)
A, U, mean_x, mean_y = fit_pca(XX, YY, best_rank)

test_trajectory = testing_data[0][0, :]
initial_condition = test_trajectory[:best_delay]
n_predict = min(2500, len(test_trajectory) - best_delay)

predicted = iterate_trajectory(initial_condition, A, mean_x, mean_y, U, n_predict)
actual = test_trajectory[:n_predict]

plt.figure(figsize=(12, 6))
time_points = np.arange(n_predict)
plt.plot(time_points, actual, label="Actual")
plt.plot(time_points, predicted, label="Predicted")
plt.title("Predicted signal vs actual signal")
plt.xlabel("Time points")
plt.ylabel("Signal")
plt.legend()
plt.savefig("PredictedVSActual.png")
plt.show()
