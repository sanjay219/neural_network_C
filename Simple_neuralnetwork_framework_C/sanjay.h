#ifndef sanjay_H_
#define sanjay_H_

#include <stddef.h>
#include <stdio.h>
#include <math.h>

#ifndef NN_MALLOC
#include <stdlib.h>
#define NN_MALLOC malloc
#endif // NN_MALLOC

#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT assert
#endif // NN_ASSERT

typedef struct
{
    size_t rows;
    size_t cols;
    size_t stride;
    float *es;
} Mat;

#define MAT_AT(m, i, j) (m).es[(i) * (m).stride + (j)]

void mat_fill(Mat m, float x);

float rand_float();

void mat_rand(Mat m, float low, float high);

Mat mat_alloc(size_t rows, size_t cols);

void mat_dot(Mat dst, Mat a, Mat b);

void mat_sum(Mat dst, Mat a);

void mat_print(Mat m, const char *name);

void mat_copy(Mat dst, Mat src);

Mat mat_row(Mat m, size_t row);

float sigmoidf(float x);

float mat_sig(Mat m);

#define MAT_PRINT(m) mat_print(m, #m)

#endif // sanjay_H_

