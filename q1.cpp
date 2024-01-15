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

#define M 256*128//2048*1024 //512*256 //1024*512
#define ARITHMETIC_OPERATIONS1 3*M
#define TIMES1 1

#define N  2048//16384 //4092 //8192
#define ARITHMETIC_OPERATIONS2 4*N*N
#define TIMES2 1


//function declaration
void initialize();
void routine1(float alpha, float beta);
void routine2(float alpha, float beta);

void routine1_vec(float alpha, float beta);
void routine2_vec(float alpha, float beta);

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

    run_time = omp_get_wtime() - start_time; //end timer
    printf("\n Time elapsed is %f secs \n %e FLOPs achieved\n", run_time, (double)(ARITHMETIC_OPERATIONS1) / ((double)run_time / TIMES1));

    printf("\nRoutine2:");
    start_time = omp_get_wtime(); //start timer

    for (t = 0; t < TIMES2; t++)
        // routine2(alpha, beta);
        routine2_vec(alpha, beta);


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

    //initialize routine1 arrays
    for (i = 0; i < N; i++) {
        x[i] = (i % 19) - 0.01f;
        w[i] = (i % 5) - 0.002f;
    }

    //initialize routine1 arrays
    for (i = 0; i < M; i++) {
        z[i] = (i % 9) - 0.08f;
        y[i] = (i % 19) + 0.07f;
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

void routine2_vec(float alpha, float beta) {
    const __m128 alpha_2 = _mm_set1_ps(alpha);
    const __m128 beta_2 = _mm_set1_ps(beta);

    unsigned int i;
    for (i = 0; i < N; ++i) {
        __m128 w2 = _mm_loadu_ps(&w[i]);
        unsigned int j;

        for (j = 0; j < N; j += 4) {
            __m128 A2 = _mm_loadu_ps(&A[i][j]);
            __m128 x2 = _mm_loadu_ps(&x[j]);

            __m128 multiple1 = _mm_mul_ps(A2, x2);
            __m128 multiple2 = _mm_mul_ps(alpha_2,multiple1);
            __m128 sub = _mm_sub_ps(w2, beta_2);

            w2 = _mm_add_ps(sub, multiple2);
        }
        _mm_storeu_ps(&w[i], w2);
    }
}