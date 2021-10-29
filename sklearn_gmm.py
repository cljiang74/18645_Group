resp = random_state.rand(n_samples, self.n_components)
resp /= resp.sum(axis=1)[:, np.newaxis]


def _estimate_gaussian_covariances_spherical(resp, X, nk, means, reg_covar):
    avg_X2 = np.dot(resp.T, X * X) / nk[:, np.newaxis]
    avg_means2 = means ** 2
    avg_X_means = means * np.dot(resp.T, X) / nk[:, np.newaxis]
    return (avg_X2 - 2 * avg_X_means + avg_means2 + reg_covar).mean(1)
    # return _estimate_gaussian_covariances_diag(resp, X, nk, means, reg_covar).mean(1)


def _estimate_gaussian_parameters(X, resp, reg_covar, covariance_type = "spherical"):
    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    means = np.dot(resp.T, X) / nk[:, np.newaxis]
    covariances = {
        "spherical": _estimate_gaussian_covariances_spherical,
    }[covariance_type](resp, X, nk, means, reg_covar)
    return nk, means, covariances


def _m_step(self, X, log_resp):
    n_samples, _ = X.shape
    self.weights_, self.means_, self.covariances_ = _estimate_gaussian_parameters(
        X, np.exp(log_resp), self.reg_covar, self.covariance_type
    )
    self.weights_ /= n_samples
    self.precisions_cholesky_ = _compute_precision_cholesky(
        self.covariances_, self.covariance_type
    )