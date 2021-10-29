from sklearn import mixture
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

Dataset = np.loadtxt('gmm_data.txt')
model = mixture.GaussianMixture(n_components=3, covariance_type='spherical', init_params='random', verbose=1)
"""
resp = random_state.rand(n_samples, self.n_components)
resp /= resp.sum(axis=1)[:, np.newaxis]
"""
model.fit(Dataset)
print('weights_pi: {}.'.format(model.weights_))
print('means: {}.'.format(model.means_))
print('covariances: {}.'.format(model.covariances_))
weights_ = model.weights_
means_ = model.means_
covariances_ = model.covariances_
pred = model.predict(Dataset)

for i, (color) in enumerate(zip(['blue', 'green', 'red'])):
    plt.scatter(Dataset[pred == i, 0], Dataset[pred == i, 1], 5, color=color)
plt.savefig('First and second dimensions plot.jpg', dpi=500)
plt.clf()
for i, (color) in enumerate(zip(['blue', 'green', 'red'])):
    plt.scatter(Dataset[pred == i, 2], Dataset[pred == i, 3], 5, color=color)
plt.savefig('Third and fourth dimensions plot.jpg', dpi=500)
plt.clf()
for i, (color) in enumerate(zip(['blue', 'green', 'red'])):
    plt.scatter(Dataset[pred == i, 3], Dataset[pred == i, 4], 5, color=color)
plt.savefig('Fourth and fifth dimensions plot.jpg', dpi=500)

mu_t = [np.random.rand(5) for _ in range(3)]
mu_t1 = [np.random.rand(5) for _ in range(3)]
while np.abs(np.mean(mu_t) - np.mean(mu_t1)) > 0.001:
    mu_t = mu_t1
    pdf = [multivariate_normal.pdf(Dataset, mean=mu_t[i], cov=covariances_[i]) for i in range(3)]
    E = [pdf[i] * weights_[i] / (sum(pdf[i] * weights_[i] for i in range(3))) for i in range(3)]
    mu_t1 = [sum(E[i][:, None]*Dataset)/sum(E[i]) for i in range(3)]
print('mu_t1: {}'.format(mu_t1))

mu_t = [np.random.rand(5) for _ in range(3)]
mu_t1 = [np.random.rand(5) for _ in range(3)]
pi_t = [np.random.rand(1) for _ in range(3)]
pi_t1 = [np.random.rand(1) for _ in range(3)]
while np.abs(np.mean(mu_t) - np.mean(mu_t1)) and np.abs(np.mean(pi_t) - np.mean(pi_t1)) > 0.001:
    mu_t = mu_t1
    pi_t = pi_t1
    pdf = [multivariate_normal.pdf(Dataset, mean=mu_t[i], cov=covariances_[i]) for i in range(3)]
    E = [pdf[i] * pi_t[i] / (sum(pdf[i] * pi_t[i] for i in range(3))) for i in range(3)]
    mu_t1 = [sum(E[i][:, None]*Dataset)/sum(E[i]) for i in range(3)]
    pi_t1 = [sum(E[i])/len(Dataset) for i in range(3)]
print('mu_t1: {}'.format(mu_t1))
print('pi_t1: {}'.format(pi_t1))
