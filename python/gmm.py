from sklearn import mixture
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import time
from sklearn_gmm import _estimate_log_gaussian_prob

# Dataset = np.loadtxt('gmm_data.txt')
Dataset = np.loadtxt('gmm_data_droped.txt')
Dataset_T = Dataset.T
model = mixture.GaussianMixture(n_components=3, covariance_type='spherical', init_params='random', verbose=1)
model.fit(Dataset)
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

total_time = np.zeros((10))
for j in range(1):
    for i in range(10):
        temp_dataset = Dataset[:i * 300 + 300] 
        start_time = time.time()
        pred, time_tmp = _estimate_log_gaussian_prob(temp_dataset, means_, precisions_cholesky_, 'spherical')
        if len(pred) == 3000:
            np.savetxt("../kernel/reference.txt", pred, fmt='%.5lf')
        pred += np.log(weights_)
        total_time[i] += time_tmp
        pred = pred.argmax(axis=1)
plt.plot(np.arange(10) * 300 + 300, total_time)
plt.savefig('total_time.jpg', dpi = 200)
plt.clf()

for i, (color) in enumerate(zip(['blue', 'green', 'red'])):
    plt.scatter(Dataset[pred == i, 0], Dataset[pred == i, 1], 5, color=color)
plt.savefig('First and second dimensions plot.jpg', dpi=500)
plt.clf()
for i, (color) in enumerate(zip(['blue', 'green', 'red'])):
    plt.scatter(Dataset[pred == i, 1], Dataset[pred == i, 2], 5, color=color)
plt.savefig('Second and third dimensions plot.jpg', dpi=500)
plt.clf()
for i, (color) in enumerate(zip(['blue', 'green', 'red'])):
    plt.scatter(Dataset[pred == i, 2], Dataset[pred == i, 3], 5, color=color)
plt.savefig('Third and fourth dimensions plot.jpg', dpi=500)