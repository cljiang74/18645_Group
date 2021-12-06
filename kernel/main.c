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
double *X_T;
double *means;
double *precisions_chol;
double time[7];
int runs = 1;

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

    double *res;
    for (int i = 0; i < runs; i++)
    {
        res = (double *)estimate_log_gaussian_prob(X, X_T, n_samples, n_features, n_components, means, precisions_chol);
    }
    // double *res = (double *)estimate_log_gaussian_prob(X, X_T, n_samples, n_features, n_components, means, precisions_chol);
    printf("Time:\n");
    for (int i = 0; i < 7; i++)
    {
        printf("%lf\n", time[i] / runs);
    }
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
    double *means_T = (double *)memalign(64, n_features * n_features * sizeof(double));                        // shape: [n_features, n_features]
    double *log_prob2_means_T_precisions = (double *)memalign(64, n_features * n_components * sizeof(double)); // shape: [n_features, n_components]
    double *log_prob2_T = (double *)memalign(64, n_components * n_samples * sizeof(double));                   // shape: [n_components, n_samples]
    double *log_prob3_einsum = (double *)memalign(64, n_samples * n_components * sizeof(double));              // shape: [n_components]
    double *log_prob3 = (double *)memalign(64, n_samples * n_components * sizeof(double));                     // shape: [n_samples, n_components]
    double *log_prob3_T = (double *)memalign(64, n_components * n_samples * sizeof(double));                   // shape: [n_samples, n_components]
    double *constant_broadcast = (double *)memalign(64, n_components * sizeof(double));                        // shape: [n_components]
    double *res = (double *)memalign(64, n_samples * n_components * sizeof(double));                           // shape: [n_samples, n_components]
    double *res_T = (double *)memalign(64, n_samples * n_components * sizeof(double));
    FILE *fp = fopen("time.txt", "w");
    unsigned long long t0, t1, t_sum;

    fprintf(fp, "performance in sequential code\n");
    fprintf(fp, "precisions = precisions_chol ** 2\n");
    __m256d precisions_temp = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
    __m256d precisions_chol_temp = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
    precisions_chol_temp = _mm256_load_pd((double *)&precisions_chol[0]);
    for (int i = 0; i < n_components; i++)
    {
        log_det[i] = n_features * log(precisions_chol[i]);
    }
    t0 = rdtsc();
    precisions_temp = _mm256_fmadd_pd(precisions_chol_temp, precisions_chol_temp, precisions_temp);
    _mm256_store_pd((double *)&precisions[0], precisions_temp);

    t1 = rdtsc();
    time[0] += t1 - t0;
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
    t0 = rdtsc();
    __m256d c_temp_1 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
    __m256d c_temp_2 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
    __m256d c_temp_3 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
    __m256d c_temp_4 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
    __m256d means_T_temp_1 = _mm256_load_pd((double *)&means_T[0]);
    __m256d means_T_temp_2 = _mm256_load_pd((double *)&means_T[4]);
    __m256d means_T_temp_3 = _mm256_load_pd((double *)&means_T[8]);
    __m256d means_T_temp_4 = _mm256_load_pd((double *)&means_T[12]);
    c_temp_1 = _mm256_fmadd_pd(means_T_temp_1, means_T_temp_1, c_temp_1);
    c_temp_2 = _mm256_fmadd_pd(means_T_temp_2, means_T_temp_2, c_temp_2);
    c_temp_3 = _mm256_fmadd_pd(means_T_temp_3, means_T_temp_3, c_temp_3);
    c_temp_4 = _mm256_fmadd_pd(means_T_temp_4, means_T_temp_4, c_temp_4);
    t1 = rdtsc();
    time[1] += t1 - t0;
    fprintf(fp, "%lld\n", t1 - t0);

    fprintf(fp, "np.sum(means ** 2, 1) * precisions\n");
    t0 = rdtsc();
    // np.sum(means ** 2, 1)
    // [n_components]
    c_temp_1 = _mm256_add_pd(c_temp_1, c_temp_2);
    c_temp_2 = _mm256_add_pd(c_temp_3, c_temp_4);
    c_temp_1 = _mm256_add_pd(c_temp_1, c_temp_2);
    c_temp_3 = _mm256_set_pd(-0.5, -0.5, -0.5, -0.5);

    // c_temp_1 == np.sum(means ** 2, 1) * precisions
    // [n_components]
    c_temp_1 = _mm256_mul_pd(c_temp_1, precisions_temp);

    t1 = rdtsc();
    time[2] += t1 - t0;
    fprintf(fp, "%lld\n", t1 - t0);

    // Precalculate constant_broadcast in the final step
    double con = -0.5 * n_features * log(2 * PI);
    c_temp_1 = _mm256_mul_pd(c_temp_1, c_temp_3);
    _mm256_store_pd((double *)log_prob1, c_temp_1);
    __m256d res_constant = _mm256_set_pd(con, con, con, con);
    __m256d constant_broadcast_simd = _mm256_load_pd((double *)log_det);
    constant_broadcast_simd = _mm256_add_pd(constant_broadcast_simd, c_temp_1);
    constant_broadcast_simd = _mm256_add_pd(constant_broadcast_simd, res_constant);
    _mm256_store_pd((double *)&constant_broadcast[0], constant_broadcast_simd);

    fprintf(fp, "2 * np.dot(X, means.T * precisions)\n");
    t0 = rdtsc();
    t_sum = 0;
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

    // // multiply by 2
    // c_temp_1 = _mm256_add_pd(c_temp_1, c_temp_1);
    // c_temp_2 = _mm256_add_pd(c_temp_2, c_temp_2);
    // c_temp_3 = _mm256_add_pd(c_temp_3, c_temp_3);

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
        t0 = rdtsc();
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
        t_sum += rdtsc() - t0;
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

    // // Naive transpose
    // for (int i = 0; i < n_samples; i++)
    // {
    //     for (int j = 0; j < n_components; j++)
    //     {
    //         log_prob2[i * n_components + j] = log_prob2_T[j * n_samples + i];
    //     }
    // }

    // // Printing verification
    // for (int i = 0; i < n_samples; i++)
    // {
    //     for (int j = 0; j < n_components; j++)
    //     {
    //         printf("%.5lf ", log_prob2[i * n_components + j]);
    //     }
    //     printf("\n");
    // }

    t1 = rdtsc();
    time[3] += t_sum;
    fprintf(fp, "%lld\n", t1 - t0);

    fprintf(fp, "np.einsum('ij,ij->i', X, X)\n");
    t0 = rdtsc();
    t_sum = 0;
    // np.einsum("ij,ij->i", X, X)
    // X column major order, use X_T
    __m256d X_T_temp_4;
    __m256d X_T_temp_5;
    __m256d X_T_temp_6;
    __m256d X_T_temp_7;
    __m256d X_T_temp_8;
//     int num_threads = 32;

// #pragma omp parallel num_threads(num_threads)
//     {
//         int id = omp_get_thread_num();
        for (int i = 0; i < n_samples; i += 32)
        // for (int i = 0 + id * (n_samples / num_threads); i < 0 + id * (n_samples / num_threads) + n_samples / num_threads; i += 32)
        {
            c_temp_1 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
            c_temp_2 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
            c_temp_3 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
            c_temp_4 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
            c_temp_5 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
            c_temp_6 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
            c_temp_7 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
            c_temp_8 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
            t0 = rdtsc();
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
            t_sum += rdtsc() - t0;
            _mm256_store_pd((double *)&log_prob3_einsum[i + 0], c_temp_1);
            _mm256_store_pd((double *)&log_prob3_einsum[i + 4], c_temp_2);
            _mm256_store_pd((double *)&log_prob3_einsum[i + 8], c_temp_3);
            _mm256_store_pd((double *)&log_prob3_einsum[i + 12], c_temp_4);
            _mm256_store_pd((double *)&log_prob3_einsum[i + 16], c_temp_5);
            _mm256_store_pd((double *)&log_prob3_einsum[i + 20], c_temp_6);
            _mm256_store_pd((double *)&log_prob3_einsum[i + 24], c_temp_7);
            _mm256_store_pd((double *)&log_prob3_einsum[i + 28], c_temp_8);
        }
    // }
    // // Handle edge case
    // c_temp_1 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
    // c_temp_2 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
    // c_temp_3 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
    // c_temp_4 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
    // c_temp_5 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
    // c_temp_6 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
    // for (int j = 0; j < n_features; j++)
    // {
    //     X_T_temp_1 = _mm256_load_pd((double *)&X_T[29984 + j * n_samples + 0]);
    //     X_T_temp_2 = _mm256_load_pd((double *)&X_T[29984 + j * n_samples + 4]);
    //     X_T_temp_3 = _mm256_load_pd((double *)&X_T[29984 + j * n_samples + 8]);
    //     X_T_temp_4 = _mm256_load_pd((double *)&X_T[29984 + j * n_samples + 12]);
    //     c_temp_1 = _mm256_fmadd_pd(X_T_temp_1, X_T_temp_1, c_temp_1);
    //     c_temp_2 = _mm256_fmadd_pd(X_T_temp_2, X_T_temp_2, c_temp_2);
    //     c_temp_3 = _mm256_fmadd_pd(X_T_temp_3, X_T_temp_3, c_temp_3);
    //     c_temp_4 = _mm256_fmadd_pd(X_T_temp_4, X_T_temp_4, c_temp_4);
    // }
    // _mm256_store_pd((double *)&log_prob3_einsum[29984], c_temp_1);
    // _mm256_store_pd((double *)&log_prob3_einsum[29988], c_temp_2);
    // _mm256_store_pd((double *)&log_prob3_einsum[29992], c_temp_3);
    // _mm256_store_pd((double *)&log_prob3_einsum[29996], c_temp_4);
    t1 = rdtsc();
    time[4] += t_sum;
    fprintf(fp, "%lld\n", t1 - t0);

    fprintf(fp, "np.outer(np.einsum('ij,ij->i', X, X), precisions)\n");
    t0 = rdtsc();
    t_sum = 0;
    // np.outer(np.einsum("ij,ij->i", X, X), precisions)
    // [n_samples] outer [n_conponents] -> [n_samples, n_components]
    // for (int i = 0; i < n_samples; i++)
    // {
    //     for (int j = 0; j < n_components; j++)
    //     {
    //         log_prob3[i * n_components + j] = log_prob3_einsum[i] * precisions[j];
    //     }
    // }

    precisions[0] *= -0.5;
    precisions[1] *= -0.5;
    precisions[2] *= -0.5;
    broad_temp1 = _mm256_broadcast_sd((double *)&precisions[0]);
    broad_temp2 = _mm256_broadcast_sd((double *)&precisions[1]);
    broad_temp3 = _mm256_broadcast_sd((double *)&precisions[2]);

    for (int i = 0; i < n_samples; i += 12)
    {
        t0 = rdtsc();
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
        t_sum += rdtsc() - t0;
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
    t1 = rdtsc();
    time[5] += t_sum;
    fprintf(fp, "%lld\n", t1 - t0);

    // Naive transpose
    for (int i = 0; i < n_samples; i++)
    {
        for (int j = 0; j < n_components; j++)
        {
            log_prob3[i * n_components + j] = log_prob3_T[j * n_samples + i];
        }
    }

    fprintf(fp, "-0.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det, end_time\n");
    t0 = rdtsc();
    t_sum = 0;

    __m256d b2_11;
    __m256d b2_12;
    __m256d b2_13;
    __m256d b2_21;
    __m256d b2_22;
    __m256d b2_23;
    __m256d b3_11;
    __m256d b3_12;
    __m256d b3_13;
    __m256d b3_21;
    __m256d b3_22;
    __m256d b3_23;
    __m256d c_broad_1 = _mm256_broadcast_sd((double *)&constant_broadcast[0]);
    __m256d c_broad_2 = _mm256_broadcast_sd((double *)&constant_broadcast[1]);
    __m256d c_broad_3 = _mm256_broadcast_sd((double *)&constant_broadcast[2]);
    for (int i = 0; i < n_samples; i += 8)
    {
        t0 = rdtsc();
        b2_11 = _mm256_load_pd((double *)&log_prob2_T[i + 0 * n_samples]);
        b2_12 = _mm256_load_pd((double *)&log_prob2_T[i + 1 * n_samples]);
        b2_13 = _mm256_load_pd((double *)&log_prob2_T[i + 2 * n_samples]);
        b2_21 = _mm256_load_pd((double *)&log_prob2_T[i + 4 + 0 * n_samples]);
        b2_22 = _mm256_load_pd((double *)&log_prob2_T[i + 4 + 1 * n_samples]);
        b2_23 = _mm256_load_pd((double *)&log_prob2_T[i + 4 + 2 * n_samples]);
        b3_11 = _mm256_load_pd((double *)&log_prob3_T[i + 0 * n_samples]);
        b3_12 = _mm256_load_pd((double *)&log_prob3_T[i + 1 * n_samples]);
        b3_13 = _mm256_load_pd((double *)&log_prob3_T[i + 2 * n_samples]);
        b3_21 = _mm256_load_pd((double *)&log_prob3_T[i + 4 + 0 * n_samples]);
        b3_22 = _mm256_load_pd((double *)&log_prob3_T[i + 4 + 1 * n_samples]);
        b3_23 = _mm256_load_pd((double *)&log_prob3_T[i + 4 + 2 * n_samples]);
        b2_11 = _mm256_add_pd(b2_11, b3_11);
        b2_12 = _mm256_add_pd(b2_12, b3_12);
        b2_13 = _mm256_add_pd(b2_13, b3_13);
        b2_21 = _mm256_add_pd(b2_21, b3_21);
        b2_22 = _mm256_add_pd(b2_22, b3_22);
        b2_23 = _mm256_add_pd(b2_23, b3_23);
        b2_11 = _mm256_add_pd(b2_11, c_broad_1);
        b2_12 = _mm256_add_pd(b2_12, c_broad_2);
        b2_13 = _mm256_add_pd(b2_13, c_broad_3);
        b2_21 = _mm256_add_pd(b2_21, c_broad_1);
        b2_22 = _mm256_add_pd(b2_22, c_broad_2);
        b2_23 = _mm256_add_pd(b2_23, c_broad_3);
        t_sum += rdtsc() - t0;
        _mm256_store_pd((double *)&res_T[i + 0 + 0 * n_samples], b2_11);
        _mm256_store_pd((double *)&res_T[i + 0 + 1 * n_samples], b2_12);
        _mm256_store_pd((double *)&res_T[i + 0 + 2 * n_samples], b2_13);
        _mm256_store_pd((double *)&res_T[i + 4 + 0 * n_samples], b2_21);
        _mm256_store_pd((double *)&res_T[i + 4 + 1 * n_samples], b2_22);
        _mm256_store_pd((double *)&res_T[i + 4 + 2 * n_samples], b2_23);
    }

    t1 = rdtsc();
    time[6] += t_sum;
    fprintf(fp, "%lld\n", t1 - t0);

    // Naive transpose
    for (int i = 0; i < n_samples; i++)
    {
        for (int j = 0; j < n_components; j++)
        {
            res[i * n_components + j] = res_T[j * n_samples + i];
        }
    }

    fclose(fp);
    return res;
}