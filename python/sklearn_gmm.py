import numpy as np
import time

def _estimate_log_gaussian_prob(X, means, precisions_chol, covariance_type = "spherical"):
    n_samples, n_features = X.shape
    log_det = n_features * (np.log(precisions_chol))
    precisions = precisions_chol ** 2
    start_time = time.time()
    kernel_1 = np.dot(X, means.T * precisions)
    precisions *= -0.5
    kernel_2 = np.einsum("ij,ij->i", X, X)
    kernel_3 = np.outer(kernel_2, precisions)
    a = kernel_1
    b = kernel_3
    c = np.sum(means ** 2, 1) * precisions + -0.5 * n_features * np.log(2 * np.pi) + log_det
    kernel_4 = a + b + c
    end_time = time.time() - start_time
    return kernel_4, end_time
