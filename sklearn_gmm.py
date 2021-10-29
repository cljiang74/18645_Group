import numpy as np

def _estimate_log_gaussian_prob(X, means, precisions_chol, covariance_type = "spherical"):
    n_samples, n_features = X.shape
    log_det = n_features * (np.log(precisions_chol))
    precisions = precisions_chol ** 2
    log_prob = (
        np.sum(means ** 2, 1) * precisions
        - 2 * np.dot(X, means.T * precisions)
        + np.outer(np.einsum("ij,ij->i", X, X), precisions)
    )
    return -0.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det
