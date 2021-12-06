#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include "immintrin.h"

#define PI 3.1415926535

int n_samples = 3000;
int n_features = 4;
int n_components = 3;

double *X;
double *X_T;
double *means;
double *precisions_chol;

void input()
{
    X = (double *)memalign(64, n_samples * n_features * sizeof(double));
    X_T = (double *)memalign(64, n_samples * n_features * sizeof(double));
    means = (double *)memalign(64, n_components * n_features * sizeof(double));
    precisions_chol = (double *)memalign(64, n_components * sizeof(double));

    for (int i = 0; i < n_samples * n_features; i++)
        scanf("%lf", &X[i]);
    for (int i = 0; i < n_samples * n_features; i++)
        scanf("%lf", &X_T[i]);
    for (int i = 0; i < n_components * n_features; i++)
        scanf("%lf", &means[i]);
    for (int i = 0; i < n_components; i++)
        scanf("%lf", &precisions_chol[i]);
}

double *estimate_log_gaussian_prob(double *X,
                                   double *X_T,
                                   int n_samples,
                                   int n_features,
                                   int n_components,
                                   double *means,
                                   double *precisions_chol);
// timing routine for reading the time stamp counter
static __inline__ unsigned long long rdtsc(void)
{
    unsigned hi, lo;
    __asm__ __volatile__("rdtsc"
                         : "=a"(lo), "=d"(hi));
    return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
}

int main(int argc, char **argv)
{
    input();

    double *res = (double *)estimate_log_gaussian_prob(X, X_T, n_samples, n_features, n_components, means, precisions_chol);
    FILE *fp;
    fp = fopen("res.txt", "w");
    for (int i = 0; i < n_samples; i++)
    {
        for (int j = 0; j < n_components; j++)
        {
            fprintf(fp, "%.5lf ", res[i * n_components + j]);
            // printf("%.5lf ", res[i * n_components + j]);
        }
        fprintf(fp, "\n");
        // printf("\n");
    }
    fclose(fp);
    return 0;
}

double *estimate_log_gaussian_prob(double *X,
                                   double *X_T,
                                   int n_samples,
                                   int n_features,
                                   int n_components,
                                   double *means,
                                   double *precisions_chol)
{
    double *log_det = (double *)memalign(64, n_components * sizeof(double));
    double *precisions = (double *)memalign(64, n_components * sizeof(double));
    double *log_prob1 = (double *)memalign(64, n_components * sizeof(double));                                 // shape: [n_components,]
    double *log_prob2 = (double *)memalign(64, n_samples * n_components * sizeof(double));                     // shape: [n_samples, n_components]
    double *log_prob3 = (double *)memalign(64, n_samples * n_components * sizeof(double));                     // shape: [n_samples, n_components]
    double *res = (double *)memalign(64, n_samples * n_components * sizeof(double));                           // shape: [n_samples, n_components]
    double *means_T = (double *)memalign(64, n_features * n_features * sizeof(double));                        // shape: [n_features, n_features], add dummy
    double *log_prob2_means_T_precisions = (double *)memalign(64, n_features * n_components * sizeof(double)); // shape: [n_features, n_components]
    double *log_prob2_T = (double *)memalign(64, n_components * n_samples * sizeof(double));                   // shape: [n_components, n_samples]
    double *log_prob3_einsum = (double *)memalign(64, n_samples * n_components * sizeof(double));              // shape: [n_components]
    double *log_prob3_T = (double *)memalign(64, n_components * n_samples * sizeof(double));                   // shape: [n_samples, n_components]
    double *res_T = (double *)memalign(64, n_samples * n_components * sizeof(double));
    FILE *fp = fopen("time.txt", "w");
    unsigned long long t0, t1;

    fprintf(fp, "performance in sequential code\n");
    fprintf(fp, "precisions = precisions_chol ** 2\n");
    __m256d precisions_temp = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
    __m256d precisions_chol_temp = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
    precisions_chol_temp = _mm256_load_pd((double *)&precisions_chol[0]);
    for (int i = 0; i < n_components; i++)
    {
        log_det[i] = n_features * log(precisions_chol[i]);
    }
    precisions_temp = _mm256_fmadd_pd(precisions_chol_temp, precisions_chol_temp, precisions_temp);
    t0 = rdtsc();
    _mm256_store_pd((double *)&precisions[0], precisions_temp);

    t1 = rdtsc();
    fprintf(fp, "%lld\n", t1 - t0);

    // mean.T
    // [n_components, n_features] -> [n_features, n_components]
    for (int i = 0; i < n_components; i++)
    {
        for (int j = 0; j < n_features; j++)
        {
            means_T[j * n_features + i] = means[i * n_features + j];
        }
    }

    fprintf(fp, "means ** 2\n");
    // means ** 2
    __m256d c_temp_1 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
    __m256d c_temp_2 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
    __m256d c_temp_3 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
    __m256d c_temp_4 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
    t0 = rdtsc();
    __m256d means_T_temp_1 = _mm256_load_pd((double *)&means_T[0]);
    __m256d means_T_temp_2 = _mm256_load_pd((double *)&means_T[4]);
    __m256d means_T_temp_3 = _mm256_load_pd((double *)&means_T[8]);
    __m256d means_T_temp_4 = _mm256_load_pd((double *)&means_T[12]);
    c_temp_1 = _mm256_fmadd_pd(means_T_temp_1, means_T_temp_1, c_temp_1);
    c_temp_2 = _mm256_fmadd_pd(means_T_temp_2, means_T_temp_2, c_temp_2);
    c_temp_3 = _mm256_fmadd_pd(means_T_temp_3, means_T_temp_3, c_temp_3);
    c_temp_4 = _mm256_fmadd_pd(means_T_temp_4, means_T_temp_4, c_temp_4);
    t1 = rdtsc();
    fprintf(fp, "%lld\n", t1 - t0);

    fprintf(fp, "np.sum(means ** 2, 1) * precisions\n");
    t0 = rdtsc();
    // np.sum(means ** 2, 1)
    // [n_components]
    c_temp_1 = _mm256_add_pd(c_temp_1, c_temp_2);
    c_temp_2 = _mm256_add_pd(c_temp_3, c_temp_4);
    c_temp_1 = _mm256_add_pd(c_temp_1, c_temp_2);
    c_temp_4 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);

    // c_temp_1 == np.sum(means ** 2, 1) * precisions
    // [n_components]
    c_temp_1 = _mm256_fmadd_pd(c_temp_1, precisions_temp, c_temp_4);
    _mm256_store_pd((double *)log_prob1, c_temp_1);

    t1 = rdtsc();
    fprintf(fp, "%lld\n", t1 - t0);

    fprintf(fp, "means.T * precisions\n");
    t0 = rdtsc();
    // means.T * precisions
    // [n_features, n_components]

    t1 = rdtsc();
    fprintf(fp, "%lld\n", t1 - t0);

    fprintf(fp, "2 * np.dot(X, means.T * precisions)\n");
    t0 = rdtsc();
    // 2 * np.dot(X, means.T * precisions)
    //[n_samples, n_features] dot [n_features, n_components] -> [n_samples, n_components]
    __m256d precisions_broadcast_1 = _mm256_broadcast_sd((double *)&precisions[0]);
    __m256d precisions_broadcast_2 = _mm256_broadcast_sd((double *)&precisions[1]);
    __m256d precisions_broadcast_3 = _mm256_broadcast_sd((double *)&precisions[2]);
    __m256d means_temp_1 = _mm256_load_pd((double *)&means[0]);
    __m256d means_temp_2 = _mm256_load_pd((double *)&means[4]);
    __m256d means_temp_3 = _mm256_load_pd((double *)&means[8]);
    // means.T * precisions
    c_temp_1 = _mm256_mul_pd(means_temp_1, precisions_broadcast_1);
    c_temp_2 = _mm256_mul_pd(means_temp_2, precisions_broadcast_2);
    c_temp_3 = _mm256_mul_pd(means_temp_3, precisions_broadcast_3);

    // multiply by 2
    c_temp_1 = _mm256_add_pd(c_temp_1, c_temp_1);
    c_temp_2 = _mm256_add_pd(c_temp_2, c_temp_2);
    c_temp_3 = _mm256_add_pd(c_temp_3, c_temp_3);

    _mm256_store_pd((double *)&log_prob2_means_T_precisions[0], c_temp_1);
    _mm256_store_pd((double *)&log_prob2_means_T_precisions[4], c_temp_2);
    _mm256_store_pd((double *)&log_prob2_means_T_precisions[8], c_temp_3);

    // np.dot(X, log_prob2_means_T_precisions) broadcast b
    // 12 x 3
    __m256d c_temp_5 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
    __m256d c_temp_6 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
    __m256d c_temp_7 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
    __m256d c_temp_8 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
    __m256d c_temp_9 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
    __m256d X_T_temp_1;
    __m256d X_T_temp_2;
    __m256d X_T_temp_3;
    __m256d broad_temp1;
    __m256d broad_temp2;
    __m256d broad_temp3;
    
    for (int i = 0; i < n_samples; i += 12)
    {
        c_temp_1 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
        c_temp_2 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
        c_temp_3 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
        c_temp_4 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
        c_temp_5 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
        c_temp_6 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
        c_temp_7 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
        c_temp_8 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
        c_temp_9 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
        for (int j = 0; j < n_features; j++)
        {
            X_T_temp_1 = _mm256_load_pd((double *)&X_T[i + j * n_samples + 0]);
            X_T_temp_2 = _mm256_load_pd((double *)&X_T[i + j * n_samples + 4]);
            X_T_temp_3 = _mm256_load_pd((double *)&X_T[i + j * n_samples + 8]);
            broad_temp1 = _mm256_broadcast_sd((double *)&log_prob2_means_T_precisions[j + 0]);
            broad_temp2 = _mm256_broadcast_sd((double *)&log_prob2_means_T_precisions[j + 4]);
            broad_temp3 = _mm256_broadcast_sd((double *)&log_prob2_means_T_precisions[j + 8]);
            c_temp_1 = _mm256_fmadd_pd(X_T_temp_1, broad_temp1, c_temp_1);
            c_temp_2 = _mm256_fmadd_pd(X_T_temp_1, broad_temp2, c_temp_2);
            c_temp_3 = _mm256_fmadd_pd(X_T_temp_1, broad_temp3, c_temp_3);
            c_temp_4 = _mm256_fmadd_pd(X_T_temp_2, broad_temp1, c_temp_4);
            c_temp_5 = _mm256_fmadd_pd(X_T_temp_2, broad_temp2, c_temp_5);
            c_temp_6 = _mm256_fmadd_pd(X_T_temp_2, broad_temp3, c_temp_6);
            c_temp_7 = _mm256_fmadd_pd(X_T_temp_3, broad_temp1, c_temp_7);
            c_temp_8 = _mm256_fmadd_pd(X_T_temp_3, broad_temp2, c_temp_8);
            c_temp_9 = _mm256_fmadd_pd(X_T_temp_3, broad_temp3, c_temp_9);
        }
        _mm256_store_pd((double *)&log_prob2_T[i + 0 * n_samples + 0], c_temp_1);
        _mm256_store_pd((double *)&log_prob2_T[i + 1 * n_samples + 0], c_temp_2);
        _mm256_store_pd((double *)&log_prob2_T[i + 2 * n_samples + 0], c_temp_3);
        _mm256_store_pd((double *)&log_prob2_T[i + 0 * n_samples + 4], c_temp_4);
        _mm256_store_pd((double *)&log_prob2_T[i + 1 * n_samples + 4], c_temp_5);
        _mm256_store_pd((double *)&log_prob2_T[i + 2 * n_samples + 4], c_temp_6);
        _mm256_store_pd((double *)&log_prob2_T[i + 0 * n_samples + 8], c_temp_7);
        _mm256_store_pd((double *)&log_prob2_T[i + 1 * n_samples + 8], c_temp_8);
        _mm256_store_pd((double *)&log_prob2_T[i + 2 * n_samples + 8], c_temp_9);
    }

    // Naive transpose
    // for (int i = 0; i < n_samples; i++)
    // {
    //     for (int j = 0; j < n_components; j++)
    //     {
    //         log_prob2[i * n_components + j] = log_prob2_T[j * n_samples + i];
    //     }
    // }

    t1 = rdtsc();
    fprintf(fp, "%lld\n", t1 - t0);

    fprintf(fp, "np.einsum('ij,ij->i', X, X)\n");
    t0 = rdtsc();
    // np.einsum("ij,ij->i", X, X)
    // X column major order, use X_T
    __m256d X_T_temp_4;
    __m256d X_T_temp_5;
    __m256d X_T_temp_6;
    __m256d X_T_temp_7;
    __m256d X_T_temp_8;
    
    for (int i = 0; i < n_samples; i += 32)
    {
        c_temp_1 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
        c_temp_2 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
        c_temp_3 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
        c_temp_4 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
        c_temp_5 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
        c_temp_6 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
        c_temp_7 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
        c_temp_8 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
        for (int j = 0; j < n_features; j++)
        {
            X_T_temp_1 = _mm256_load_pd((double *)&X_T[i + j * n_samples + 0]);
            X_T_temp_2 = _mm256_load_pd((double *)&X_T[i + j * n_samples + 4]);
            X_T_temp_3 = _mm256_load_pd((double *)&X_T[i + j * n_samples + 8]);
            X_T_temp_4 = _mm256_load_pd((double *)&X_T[i + j * n_samples + 12]);
            X_T_temp_5 = _mm256_load_pd((double *)&X_T[i + j * n_samples + 16]);
            X_T_temp_6 = _mm256_load_pd((double *)&X_T[i + j * n_samples + 20]);
            X_T_temp_7 = _mm256_load_pd((double *)&X_T[i + j * n_samples + 24]);
            X_T_temp_8 = _mm256_load_pd((double *)&X_T[i + j * n_samples + 28]);
            c_temp_1 = _mm256_fmadd_pd(X_T_temp_1, X_T_temp_1, c_temp_1);
            c_temp_2 = _mm256_fmadd_pd(X_T_temp_2, X_T_temp_2, c_temp_2);
            c_temp_3 = _mm256_fmadd_pd(X_T_temp_3, X_T_temp_3, c_temp_3);
            c_temp_4 = _mm256_fmadd_pd(X_T_temp_4, X_T_temp_4, c_temp_4);
            c_temp_5 = _mm256_fmadd_pd(X_T_temp_5, X_T_temp_5, c_temp_5);
            c_temp_6 = _mm256_fmadd_pd(X_T_temp_6, X_T_temp_6, c_temp_6);
            c_temp_7 = _mm256_fmadd_pd(X_T_temp_7, X_T_temp_7, c_temp_7);
            c_temp_8 = _mm256_fmadd_pd(X_T_temp_8, X_T_temp_8, c_temp_8);
        }
        _mm256_store_pd((double *)&log_prob3_einsum[i + 0], c_temp_1);
        _mm256_store_pd((double *)&log_prob3_einsum[i + 4], c_temp_2);
        _mm256_store_pd((double *)&log_prob3_einsum[i + 8], c_temp_3);
        _mm256_store_pd((double *)&log_prob3_einsum[i + 12], c_temp_4);
        _mm256_store_pd((double *)&log_prob3_einsum[i + 16], c_temp_5);
        _mm256_store_pd((double *)&log_prob3_einsum[i + 20], c_temp_6);
        _mm256_store_pd((double *)&log_prob3_einsum[i + 24], c_temp_7);
        _mm256_store_pd((double *)&log_prob3_einsum[i + 28], c_temp_8);
    }
    // Handle edge case
    c_temp_1 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
    c_temp_2 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
    c_temp_3 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
    c_temp_4 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
    c_temp_5 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
    c_temp_6 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
    for (int j = 0; j < n_features; j++)
    {
        X_T_temp_1 = _mm256_load_pd((double *)&X_T[2976 + j * n_samples + 0]);
        X_T_temp_2 = _mm256_load_pd((double *)&X_T[2976 + j * n_samples + 4]);
        X_T_temp_3 = _mm256_load_pd((double *)&X_T[2976 + j * n_samples + 8]);
        X_T_temp_4 = _mm256_load_pd((double *)&X_T[2976 + j * n_samples + 12]);
        X_T_temp_5 = _mm256_load_pd((double *)&X_T[2976 + j * n_samples + 16]);
        X_T_temp_6 = _mm256_load_pd((double *)&X_T[2976 + j * n_samples + 20]);
        c_temp_1 = _mm256_fmadd_pd(X_T_temp_1, X_T_temp_1, c_temp_1);
        c_temp_2 = _mm256_fmadd_pd(X_T_temp_2, X_T_temp_2, c_temp_2);
        c_temp_3 = _mm256_fmadd_pd(X_T_temp_3, X_T_temp_3, c_temp_3);
        c_temp_4 = _mm256_fmadd_pd(X_T_temp_4, X_T_temp_4, c_temp_4);
        c_temp_5 = _mm256_fmadd_pd(X_T_temp_5, X_T_temp_5, c_temp_5);
        c_temp_6 = _mm256_fmadd_pd(X_T_temp_6, X_T_temp_6, c_temp_6);
    }
    _mm256_store_pd((double *)&log_prob3_einsum[2976], c_temp_1);
    _mm256_store_pd((double *)&log_prob3_einsum[2980], c_temp_2);
    _mm256_store_pd((double *)&log_prob3_einsum[2984], c_temp_3);
    _mm256_store_pd((double *)&log_prob3_einsum[2988], c_temp_4);
    _mm256_store_pd((double *)&log_prob3_einsum[2992], c_temp_5);
    _mm256_store_pd((double *)&log_prob3_einsum[2996], c_temp_6);
    t1 = rdtsc();
    fprintf(fp, "%lld\n", t1 - t0);

    fprintf(fp, "np.outer(np.einsum('ij,ij->i', X, X), precisions)\n");
    t0 = rdtsc();
    // np.outer(np.einsum("ij,ij->i", X, X), precisions)
    // [n_samples] outer [n_conponents] -> [n_samples, n_components]
    // for (int i = 0; i < n_samples; i++)
    // {
    //     for (int j = 0; j < n_components; j++)
    //     {
    //         log_prob3[i * n_components + j] = log_prob3_einsum[i] * precisions[j];
    //     }
    // }

    broad_temp1 = _mm256_broadcast_sd((double *)&precisions[0]);
    broad_temp2 = _mm256_broadcast_sd((double *)&precisions[1]);
    broad_temp3 = _mm256_broadcast_sd((double *)&precisions[2]);
    //  #pragma omp parallel for num_threads(4)
    for (int i = 0; i < n_samples; i += 12)
    {
        __m256d log_prob3_einsum_temp1 = _mm256_load_pd((double *)&log_prob3_einsum[i]);
        __m256d log_prob3_einsum_temp2 = _mm256_load_pd((double *)&log_prob3_einsum[i + 4]);
        __m256d log_prob3_einsum_temp3 = _mm256_load_pd((double *)&log_prob3_einsum[i + 8]);
        __m256d log_prob3_temp1 = _mm256_mul_pd(broad_temp1, log_prob3_einsum_temp1);
        __m256d log_prob3_temp2 = _mm256_mul_pd(broad_temp1, log_prob3_einsum_temp2);
        __m256d log_prob3_temp3 = _mm256_mul_pd(broad_temp1, log_prob3_einsum_temp3);
        __m256d log_prob3_temp4 = _mm256_mul_pd(broad_temp2, log_prob3_einsum_temp1);
        __m256d log_prob3_temp5 = _mm256_mul_pd(broad_temp2, log_prob3_einsum_temp2);
        __m256d log_prob3_temp6 = _mm256_mul_pd(broad_temp2, log_prob3_einsum_temp3);
        __m256d log_prob3_temp7 = _mm256_mul_pd(broad_temp3, log_prob3_einsum_temp1);
        __m256d log_prob3_temp8 = _mm256_mul_pd(broad_temp3, log_prob3_einsum_temp2);
        __m256d log_prob3_temp9 = _mm256_mul_pd(broad_temp3, log_prob3_einsum_temp3);
        _mm256_store_pd((double *)&log_prob3_T[i], log_prob3_temp1);
        _mm256_store_pd((double *)&log_prob3_T[i + 4], log_prob3_temp2);
        _mm256_store_pd((double *)&log_prob3_T[i + 8], log_prob3_temp3);
        _mm256_store_pd((double *)&log_prob3_T[i + n_samples], log_prob3_temp4);
        _mm256_store_pd((double *)&log_prob3_T[i + 4 + n_samples], log_prob3_temp5);
        _mm256_store_pd((double *)&log_prob3_T[i + 8 + n_samples], log_prob3_temp6);
        _mm256_store_pd((double *)&log_prob3_T[i + 2 * n_samples], log_prob3_temp7);
        _mm256_store_pd((double *)&log_prob3_T[i + 4 + 2 * n_samples], log_prob3_temp8);
        _mm256_store_pd((double *)&log_prob3_T[i + 8 + 2 * n_samples], log_prob3_temp9);
    }

    // Naive transpose
    // for (int i = 0; i < n_samples; i++)
    // {
    //     for (int j = 0; j < n_components; j++)
    //     {
    //         log_prob3[i * n_components + j] = log_prob3_T[j * n_samples + i];
    //     }
    // }
    t1 = rdtsc();
    fprintf(fp, "%lld\n", t1 - t0);

    fprintf(fp, "-0.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det, end_time\n");
    t0 = rdtsc();
    // // -0.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det, end_time
    // for (int i = 0; i < n_samples; i++)
    // {
    //     for (int j = 0; j < n_components; j++)
    //     {
    //         res[i * n_components + j] = -0.5 * (n_features * log(2 * PI) + log_prob1[j] -
    //                                             log_prob2_T[j * n_samples + i] + log_prob3_T[j * n_samples + i]) +
    //                                     log_det[j];
    //     }
    // }

    double con = n_features * log(2 * PI);
    __m256d res_constant = _mm256_set_pd(con, con, con, con);
    __m256d log_prob1_temp1 = _mm256_broadcast_sd((double *)&log_prob1[0]);
    __m256d log_prob1_temp2 = _mm256_broadcast_sd((double *)&log_prob1[1]);
    __m256d log_prob1_temp3 = _mm256_broadcast_sd((double *)&log_prob1[2]);
    __m256d log_log_det_temp1 = _mm256_broadcast_sd((double *)&log_det[0]);
    __m256d log_log_det_temp2 = _mm256_broadcast_sd((double *)&log_det[1]);
    __m256d log_log_det_temp3 = _mm256_broadcast_sd((double *)&log_det[2]);
    __m256d res_constant2 = _mm256_set_pd(-0.5, -0.5, -0.5, -0.5);
    for (int i = 0; i < n_samples; i += 4)
    {
        __m256d log_prob2_T_temp1 = _mm256_load_pd((double *)&log_prob2_T[i]);
        __m256d log_prob2_T_temp2 = _mm256_load_pd((double *)&log_prob2_T[i + n_samples]);
        __m256d log_prob2_T_temp3 = _mm256_load_pd((double *)&log_prob2_T[i + 2 * n_samples]);
        __m256d log_prob3_T_temp1 = _mm256_load_pd((double *)&log_prob3_T[i]);
        __m256d log_prob3_T_temp2 = _mm256_load_pd((double *)&log_prob3_T[i + n_samples]);
        __m256d log_prob3_T_temp3 = _mm256_load_pd((double *)&log_prob3_T[i + 2 * n_samples]);
        __m256d res_temp1 = _mm256_add_pd(res_constant, log_prob1_temp1);
        __m256d res_temp2 = _mm256_add_pd(res_constant, log_prob1_temp2);
        __m256d res_temp3 = _mm256_add_pd(res_constant, log_prob1_temp3);
        res_temp1 = _mm256_sub_pd(res_temp1, log_prob2_T_temp1);
        res_temp2 = _mm256_sub_pd(res_temp2, log_prob2_T_temp2);
        res_temp3 = _mm256_sub_pd(res_temp3, log_prob2_T_temp3);
        res_temp1 = _mm256_add_pd(res_temp1, log_prob3_T_temp1);
        res_temp2 = _mm256_add_pd(res_temp2, log_prob3_T_temp2);
        res_temp3 = _mm256_add_pd(res_temp3, log_prob3_T_temp3);
        res_temp1 = _mm256_mul_pd(res_temp1, res_constant2);
        res_temp2 = _mm256_mul_pd(res_temp2, res_constant2);
        res_temp3 = _mm256_mul_pd(res_temp3, res_constant2);
        res_temp1 = _mm256_add_pd(res_temp1, log_log_det_temp1);
        res_temp2 = _mm256_add_pd(res_temp2, log_log_det_temp2);
        res_temp3 = _mm256_add_pd(res_temp3, log_log_det_temp3);
        _mm256_store_pd((double *)&res_T[i], res_temp1);
        _mm256_store_pd((double *)&res_T[i + n_samples], res_temp2);
        _mm256_store_pd((double *)&res_T[i + 2 * n_samples], res_temp3);
    }

    // Naive transpose
    for (int i = 0; i < n_samples; i++)
    {
        for (int j = 0; j < n_components; j++)
        {
            res[i * n_components + j] = res_T[j * n_samples + i];
        }
    }

    t1 = rdtsc();
    fprintf(fp, "%lld\n", t1 - t0);

    fclose(fp);
    return res;
}