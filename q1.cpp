/*
------------------DR VASILIOS KELEFOURAS-----------------------------------------------------
------------------COMP1001 ------------------------------------------------------------------
------------------COMPUTER SYSTEMS MODULE-------------------------------------------------
------------------UNIVERSITY OF PLYMOUTH, SCHOOL OF ENGINEERING, COMPUTING AND MATHEMATICS---
*/


#include <stdio.h>
#include <time.h>
#include <pmmintrin.h>
#include <process.h>
#include <chrono>
#include <iostream>
#include <immintrin.h>
#include <omp.h>
#include <emmintrin.h>

//matrix and vector operations using SSE intrinsics

#define M 1024*512   //256*128//2048*1024 //512*256
#define ARITHMETIC_OPERATIONS1 3*M
#define TIMES1 1

#define N  8192  //2048//16384 //4092
#define ARITHMETIC_OPERATIONS2 4*N*N
#define TIMES2 1


constexpr auto EPSILON =0.0001;

//function declaration
void initialize();
void routine1(float alpha, float beta);
void routine2(float alpha, float beta);

unsigned short int equal(float a, float b);
int routine1_test(float, float);
int routine2_test(float, float);

void routine1_vec(float alpha, float beta);
void routine2_vec(float alpha, float beta);

__declspec(align(64)) float  test1[M];
__declspec(align(64)) float  test2[N];
__declspec(align(64)) float  y[M], z[M] ;
__declspec(align(64)) float A[N][N], x[N], w[N];

int main() {

    float alpha = 0.023f, beta = 0.045f;
    double run_time, start_time;
    unsigned int t;

    initialize();

    printf("\nRoutine1:");
    start_time = omp_get_wtime(); //start timer

    for (t = 0; t < TIMES1; t++)
        // routine1(alpha, beta);
        routine1_vec(alpha, beta);
    routine1_test(alpha, beta);

    run_time = omp_get_wtime() - start_time; //end timer
    printf("\n Time elapsed is %f secs \n %e FLOPs achieved\n", run_time, (double)(ARITHMETIC_OPERATIONS1) / ((double)run_time / TIMES1));

    printf("\nRoutine2:");
    start_time = omp_get_wtime(); //start timer

    for (t = 0; t < TIMES2; t++){
        // routine2(alpha, beta);
        routine2_vec(alpha, beta);
        routine2_test(alpha, beta);
    }

    run_time = omp_get_wtime() - start_time; //end timer
    printf("\n Time elapsed is %f secs \n %e FLOPs achieved\n", run_time, (double)(ARITHMETIC_OPERATIONS2) / ((double)run_time / TIMES2));

    return 0;
}

void initialize() {

    unsigned int i, j;

    //initialize routine2 arrays
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++) {
            A[i][j] = (i % 99) + (j % 14) + 0.013f;
        }

    //initialize routine2 arrays
    for (i = 0; i < N; i++) {
        x[i] = (i % 19) - 0.01f;
        w[i] = (i % 5) - 0.002f;
        test2[i]= (i % 5) - 0.002f;
    }

    //initialize routine1 arrays
    for (i = 0; i < M; i++) {
        z[i] = (i % 9) - 0.08f;
        y[i] = (i % 19) + 0.07f;
        test1[i] = (i % 19) + 0.07f;
    }
}

void routine1(float alpha, float beta) {

    unsigned int i;

    for (i = 0; i < M; i++)
        y[i] = alpha * y[i] + beta * z[i];

}

void routine2(float alpha, float beta) {

    unsigned int i, j;

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            w[i] = w[i] - beta + alpha * A[i][j] * x[j];

}

//Routine1 Vectorized version
void routine1_vec(float alpha, float beta) {
    const __m128 alpha1 = _mm_set1_ps(alpha);
    const __m128 beta1= _mm_set1_ps(beta);

    unsigned int i;
    for (i = 0; i < M; i += 4) {
        __m128 y1 = _mm_loadu_ps(&y[i]);
        __m128 z1 = _mm_loadu_ps(&z[i]);

        __m128 multiple1 = _mm_mul_ps(beta1, z1);
        __m128 multiple2 = _mm_mul_ps(alpha1, y1);

        y1 = _mm_add_ps(multiple1, multiple2);

        _mm_storeu_ps(&y[i], y1);
    }
}

//Routine2 Vectorized version
void routine2_vec(float alpha, float beta) {
    __m128 alpha2 = _mm_set1_ps(alpha);
    __m128 beta_n = _mm_set1_ps(N * beta);
    int assert(N % 4 == 0);
    for (unsigned i = 0; i < N; i += 4) {
        __m128 dot4_1 = _mm_setzero_ps();
        __m128 dot4_2 = dot4_1, dot4_3 = dot4_1, dot4_4 = dot4_1;
        for (unsigned j = 0; j < N; j += 4) {
            __m128 aij = _mm_loadu_ps(&A[i][j]);
            __m128 xj = _mm_loadu_ps(&x[j]);
            __m128 mres = _mm_mul_ps(aij, xj);
            dot4_1 = _mm_add_ps(dot4_1, mres);

            aij = _mm_loadu_ps(&A[i + 1][j]);
            mres = _mm_mul_ps(aij, xj);
            dot4_2 = _mm_add_ps(dot4_2, mres);

            aij = _mm_loadu_ps(&A[i + 2][j]);
            mres = _mm_mul_ps(aij, xj);
            dot4_3 = _mm_add_ps(dot4_3, mres);

            aij = _mm_loadu_ps(&A[i + 3][j]);
            mres = _mm_mul_ps(aij, xj);
            dot4_4 = _mm_add_ps(dot4_4, mres);
        }
        /* Reduce to one vector for 4 i-values */
        _MM_TRANSPOSE4_PS(dot4_1, dot4_2, dot4_3, dot4_4);
        dot4_1 = _mm_add_ps(dot4_1, dot4_2);
        dot4_3 = _mm_add_ps(dot4_3, dot4_4);
        dot4_1 = _mm_add_ps(dot4_1, dot4_3);

        dot4_1 = _mm_mul_ps(dot4_1, alpha2);
        dot4_1 = _mm_sub_ps(dot4_1, beta_n);
        __m128 wi = _mm_loadu_ps(&w[i]);
        wi = _mm_add_ps(wi, dot4_1);
        _mm_storeu_ps(&w[i], wi);
    }
}

//function to test the results of Routine 1
int routine1_test(float alpha, float beta) {
    unsigned int i,j;
    for (i = 0; i < M; i++){
        test1[i] = alpha * test1[i] + beta * z[i];
    }

    // compare

    for (j = 0; j < M; j++) {
        if (equal(y[j], test1[j]) == 1) {
            printf("\n The result of y[%d] is not equal to test1[%d]  \n", j, j);
            return 1;
        }
    }

    return 0;
}

//function to test the results of Routine 2
int routine2_test(float alpha, float beta) {
    unsigned int i, j;

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            test2[i] = test2[i] - beta + alpha * A[i][j] * x[j];

        }
    }

    // compare
    for (j = 0; j < N; j++) {
        if (equal(w[j], test2[j]) == 1) {
            printf("\n The result of w[%d] is not equal to test2[%d]  \n", j, j);
            return 1;
        }
    }

    return 0;
}


unsigned short int equal(float a, float b) {
    float temp = a - b;
    //printf("\n %f  %f", a, b);
    if ((fabs(temp) / fabs(b)) < EPSILON)
        return 0; //success
    else
        return 1; //wrong result
}