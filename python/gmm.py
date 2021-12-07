from sklearn import mixture
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import time
from sklearn_gmm import _estimate_log_gaussian_prob

data_origin = np.loadtxt('gmm_data_droped.txt')
Dataset = np.tile(data_origin, (100, 1))
Dataset_T = Dataset.T
model = mixture.GaussianMixture(n_components=3, covariance_type='spherical', init_params='random', verbose=1)
model.fit(data_origin)
predictions = model.predict(data_origin)
weights_ = model.weights_
means_ = model.means_
covariances_ = model.covariances_
precisions_cholesky_ = model.precisions_cholesky_

# Save numpy arrays to txt files for c kernel to read
with open("../kernel/dataset.txt", "w") as f:
    for i in range(len(Dataset)):
        for j in range(len(Dataset[0])):
            f.write(str(Dataset[i, j]) + "\n")

with open("../kernel/dataset_T.txt", "w") as f:
    for i in range(len(Dataset_T)):
        for j in range(len(Dataset_T[0])):
            f.write(str(Dataset_T[i, j]) + "\n")

with open("../kernel/means.txt", "w") as f:
    for i in range(len(means_)):
        for j in range(len(means_[0])):
            f.write(str(means_[i, j]) + "\n")

with open("../kernel/precisions_cholesky.txt", "w") as f:
    for i in range(len(precisions_cholesky_)):
            f.write(str(precisions_cholesky_[i]) + "\n")

runs = 100
total_time = np.zeros((runs))
for i in range(runs):
    start_time = time.time()
    pred, time_tmp = _estimate_log_gaussian_prob(Dataset, means_, precisions_cholesky_, 'spherical')
    time_tmp = time.time() - start_time
    if i == 0:
        np.savetxt("../kernel/reference.txt", pred, fmt='%.5lf')
    pred += np.log(weights_)
    total_time[i] = time_tmp
    pred = pred.argmax(axis=1)
print(np.mean(total_time))

for i, (color) in enumerate(zip(['blue', 'green', 'red'])):
    plt.scatter(data_origin[predictions == i, 0], data_origin[predictions == i, 1], 5, color=color)
plt.savefig('First and second dimensions plot.jpg', dpi=500)
plt.clf()
for i, (color) in enumerate(zip(['blue', 'green', 'red'])):
    plt.scatter(data_origin[predictions == i, 1], data_origin[predictions == i, 2], 5, color=color)
plt.savefig('Second and third dimensions plot.jpg', dpi=500)
plt.clf()
for i, (color) in enumerate(zip(['blue', 'green', 'red'])):
    plt.scatter(data_origin[predictions == i, 2], data_origin[predictions == i, 3], 5, color=color)
plt.savefig('Third and fourth dimensions plot.jpg', dpi=500)
