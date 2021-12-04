#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "immintrin.h"

#define PI 3.1415926535

double* _estimate_log_gaussian_prob(double *X,
                                    double *X_T,
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
    double *log_prob3_einsum = (double*) memalign(64, n_features * n_components * sizeof(double)); // shape: [n_components]
    
    for(int i = 0; i < n_components; i++) {
        log_det[i] = n_features * log(precisions_chol[i]);
        precisions[i] = precisions_chol[i] * precisions_chol[i]; // SIMD
        // means ** 2: FMA
        // np.sum(means ** 2, 1): MPI_Reduce [n_components, n_features] -> [n_components,]
    
    }

    // means ** 2
    for(int i = 0; i < n_components * n_features; i++) {
        log_prob1_means_sq[i] = means[i] * means[i];
    }

    // np.sum(means ** 2, 1) * precisions
    // [n_components]
    for(int i = 0; i < n_components; i++) {
        for (int j = 0; i < n_features; j++) {
            log_prob1[i] += log_prob1_means_sq[i * n_features + j];
        }
        log_prob1[i] *= precisions[i];
    }
    
    // means.T * precisions
    for (int i = 0; i < n_components; i++) {
        for (int j = 0; j < n_features; j++) {
            log_prob2_means_T_precisions[i + j * n_components] = means[j + i * n_features] * precisions[i];
        }
    }

    // 2 * np.dot(X, means.T * precisions)
    //[n_samples, n_features] dot [n_features, n_components] -> [n_samples, n_components]
    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_components; j++) {
            for (int k = 0; k < n_features; k++) {
                log_prob2[i * n_components + j] += 2 * X[i * n_features + k] * log_prob2_means_T_precisions[k * n_components + j];
            }
        }
    }
    
    // np.einsum("ij,ij->i", X, X) 
    for (int i=0; i<n_samples; i++) {
        int row_sum = 0;
        for (int j=0; j<n_features; j++) {
            row_sum += X[j + i * n_features];
        }
        log_prob3_einsum[i] = row_sum;
    }

    // np.outer(np.einsum("ij,ij->i", X, X), precisions)
    // [n_samples] outer [n_conponents] -> [n_samples, n_components]
    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_components; j++) {
            log_prob3[i * n_components + j] =  log_prob3_einsum[i] * precisions[j];
        }
    }

    // -0.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det, end_time
    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_components; j++) {
            res[i * n_components + j] = -0.5 * (n_features * log(2 * PI) + log_prob1[j] - 
                                        log_prob2[i * n_components + j] + log_prob3[i * n_components + j])+ log_det[j];
        }
    }

    return res;
}