#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include "immintrin.h"

#define PI 3.1415926535

int n_samples = 300000;
int n_features = 4;
int n_components = 3;

double *X;
double *means;
double *precisions_chol;

void input() {
    X = (double*)memalign(64, n_samples * n_features * sizeof(double));
    means = (double*)memalign(64, n_components * n_features * sizeof(double));
    precisions_chol = (double*)memalign(64, n_components * sizeof(double));

    for(int i = 0; i<n_samples * n_features; i++) scanf("%lf", &X[i]);
    for(int i = 0; i<n_components * n_features; i++) scanf("%lf", &means[i]);
    for(int i = 0; i<n_components; i++) scanf("%lf", &precisions_chol[i]);
}

double* estimate_log_gaussian_prob(double *X,
                                   int n_samples,
                                   int n_features,
                                   int n_components,
                                   double *means, 
                                   double *precisions_chol);
//timing routine for reading the time stamp counter
static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}

int main(int argc, char **argv)
{
    input();
    
    double *res = (double *)estimate_log_gaussian_prob(X, n_samples, n_features, n_components, means, precisions_chol);
    FILE* fp;
    fp = fopen("res.txt", "w");
    for(int i = 0; i<n_samples; i++){
        for (int j =0; j < n_components; j++){
            fprintf(fp,"%.5lf ", res[i * n_components + j]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
    return 0;
}


double* estimate_log_gaussian_prob(double *X,
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
    double *log_prob3_einsum = (double*) memalign(64, n_samples * n_components * sizeof(double)); // shape: [n_components]
    FILE* fp = fopen("time.txt", "w");
    unsigned long long t0, t1;
    for(int i = 0; i < n_components; i++) {
        log_det[i] = n_features * log(precisions_chol[i]);
        precisions[i] = precisions_chol[i] * precisions_chol[i]; // SIMD
        // means ** 2: FMA
        // np.sum(means ** 2, 1): MPI_Reduce [n_components, n_features] -> [n_components,]
    
    }

    fprintf(fp, "performance in sequential code\n");
    fprintf(fp, "means ** 2\n");
    // means ** 2
    t0 = rdtsc();
    for(int i = 0; i < n_components * n_features; i++) {
        log_prob1_means_sq[i] = means[i] * means[i];
    }
    t1 = rdtsc();
    fprintf(fp, "%lld\n", t1 - t0);

    fprintf(fp, "np.sum(means ** 2, 1) * precisions\n");
    t0 = rdtsc();
    // np.sum(means ** 2, 1) * precisions
    // [n_components]
    for(int i = 0; i < n_components; i++) {
        for (int j = 0; j < n_features; j++) {
            log_prob1[i] += log_prob1_means_sq[i * n_features + j];
        }
        log_prob1[i] *= precisions[i];
    }
    t1 = rdtsc();
    fprintf(fp, "%lld\n", t1 - t0);

    fprintf(fp, "means.T * precisions\n");
    t0 = rdtsc();
    // means.T * precisions
    // [n_features, n_components]
    for (int i = 0; i < n_components; i++) {
        for (int j = 0; j < n_features; j++) {
            log_prob2_means_T_precisions[i + j * n_components] = means[j + i * n_features] * precisions[i];
        }
    }
    t1 = rdtsc();
    fprintf(fp, "%lld\n", t1 - t0);

    fprintf(fp, "2 * np.dot(X, means.T * precisions)\n");
    t0 = rdtsc();
    // 2 * np.dot(X, means.T * precisions)
    //[n_samples, n_features] dot [n_features, n_components] -> [n_samples, n_components]
    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_components; j++) {
            for (int k = 0; k < n_features; k++) {
                log_prob2[i * n_components + j] += 2 * X[i * n_features + k] * log_prob2_means_T_precisions[k * n_components + j];
            }
        }
    }
    t1 = rdtsc();
    fprintf(fp, "%lld\n", t1 - t0);


    fprintf(fp, "np.einsum('ij,ij->i', X, X)\n");
    t0 = rdtsc();
    // np.einsum("ij,ij->i", X, X)
    // # pragma omp parallel for num_threads(1)
    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_features; j++) {
            log_prob3_einsum[i] += X[i * n_features + j] * X[i * n_features + j];
        }
    }
    t1 = rdtsc();
    fprintf(fp, "%lld\n", t1 - t0);

    fprintf(fp, "np.outer(np.einsum('ij,ij->i', X, X), precisions)\n");
    t0 = rdtsc();
    // np.outer(np.einsum("ij,ij->i", X, X), precisions)
    // [n_samples] outer [n_conponents] -> [n_samples, n_components]
    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_components; j++) {
            log_prob3[i * n_components + j] =  log_prob3_einsum[i] * precisions[j];
        }
    }
    t1 = rdtsc();
    fprintf(fp, "%lld\n", t1 - t0);

    fprintf(fp, "-0.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det, end_time\n");
    t0 = rdtsc();
    // -0.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det, end_time
    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_components; j++) {
            res[i * n_components + j] = -0.5 * (n_features * log(2 * PI) + log_prob1[j] - 
                                        log_prob2[i * n_components + j] + log_prob3[i * n_components + j])+ log_det[j];
        }
    }
    t1 = rdtsc();
    fprintf(fp, "%lld\n", t1 - t0);

    fclose(fp);
    return res;
}