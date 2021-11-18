#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "immintrin.h"

double* _estimate_log_gaussian_prob(double *X,
                                   int n_samples,
                                   int n_features,
                                   int n_components,
                                   double *means, 
                                   double *precisions_chol)
{   
    double *log_det = (double*) memalign(64, n_components * sizeof(double));
    double *precisions = (double*) memalign(64, n_components * sizeof(double));
    double *log_prob1 = (double*) memalign(64, n_components * sizeof(double)); // shape: [n_components,]
    double *log_prob2 = (double*) memalign(64, n_samples * n_components * sizeof(double)); // shape: [n_samples, n_components]
    double *log_prob3 = (double*) memalign(64, n_samples * n_components * sizeof(double)); // shape: [n_samples, n_components]
    double *res = (double*) memalign(64, n_samples * n_components * sizeof(double)); // shape: [n_samples, n_components]
    double *log_prob1_np_sum = (double*) memalign(64, n_components * sizeof(double)); // shape: [n_components]
    double *log_prob1_means_sq = (double*) memalign(64, n_components * n_features * sizeof(double)); // shape: [n_components, n_features]
    double *log_prob2_means_T_precisions = (double*) memalign(64, n_features * n_components * sizeof(double)); // shape: [n_features, n_components]
    
    for(int i = 0; i < n_components; i++) {
        log_det[i] = n_features * log(precisions_chol[i]);
        precisions[i] = precisions_chol[i] * precisions_chol[i]; // SIMD
        // means ** 2: FMA
        // np.sum(means ** 2, 1): MPI_Reduce [n_components, n_features] -> [n_components,]
    
    }

    for(int i = 0; i < n_components * n_features; i++) {
        log_prob1_means_sq[i] = means[i] * means[i];
    }
    for(int i = 0; i < n_components; i++) {
        for (int j = 0; i < n_features; j++) {
            log_prob1[i] += log_prob1_means_sq[i * n_features + j];
        }
        log_prob1[i] *= precisions[i];
        
    }
    for (int i = 0; i < n_components; i++) {
        for (int j = 0; j < n_features, j++) {
            log_prob2_means_T_precisions[i + j * n_components] = means[j + i * n_features] * precisions[i];
        }
    }

    return res;
}

// def _estimate_log_gaussian_prob(X, means, precisions_chol, covariance_type = "spherical"):
//     n_samples, n_features = X.shape
//     log_det = n_features * (np.log(precisions_chol))
//     precisions = precisions_chol ** 2
//     start_time = time.time()
//     end_time = time.time() - start_time
//     log_prob = (
//         np.sum(means ** 2, 1) * precisions
//         - 2 * np.dot(X, means.T * precisions)
//         + np.outer(np.einsum("ij,ij->i", X, X), precisions)
//     )
//     return -0.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det, end_time