#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float* fill_matrix_a(const int n)
{
    float* matrix = calloc(n * n, sizeof(float));
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            matrix[i * n + j] = (float)(rand() % n);
    }
    return matrix;
}

void transporation_matrix(float* matrix, const int n)
{
    for (int i = 0; i < n; i++)
        for (int j = i; j < n; j++)
        {
            matrix[j * n + i] += matrix[i * n + j];
            matrix[i * n + j] = matrix[j * n + i] - matrix[i * n + j];
            matrix[j * n + i] -= matrix[i * n + j];
        }
}

float* fill_unit_matrix(const int n)
{
    float* unit_matrix = calloc(n * n, sizeof(float));
    for (int i = 0; i < n; i++)
    {
        unit_matrix[i * n + i] = (float)1;
    }
    return unit_matrix;
}

float max_row(const float* matrix, const int n)
{
    float tmp = 0;
    float s = 0;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            s += matrix[i * n + j];
        }
        if (s > tmp)
            tmp = s;
        s = 0;
    }
    return tmp;
}

float max_column(const float* matrix, const int n)
{
    float tmp = 0;
    float s = 0;
    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < n; i++)
        {
            s += matrix[i * n + j];
        }
        if (s > tmp)
            tmp = s;
        s = 0;
    }
    return tmp;
}

float* fill_matrix_b(float* matrix, const int n)
{
    float* tmp = calloc(n * n, sizeof(float));
    float row = max_row(matrix, n);
    float column = max_column(matrix, n);
    transporation_matrix(matrix, n);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            tmp[i * n + j] = matrix[i * n + j] / (row * column);
        }
    }
    return tmp;
}

float* matrix_multiplication(const float* matrix_a, const float* matrix_b, const int
n)
{
    float* matrix_c = calloc(n * n, sizeof(float));
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int k = 0; k < n; k++)
                matrix_c[i * n + j] = matrix_a[i * n + k] * matrix_b[k * n + j];
        }
    }
    return matrix_c;
}

float* get_matrix_r(const float* unit_matrix, float* matrix_a, float* matrix_b, const
int n)
{
    float* matrix_r = calloc(n * n, sizeof(float));
    float* tmp = matrix_multiplication(matrix_b, matrix_a, n);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            matrix_r[i * n + j] = unit_matrix[i * n + j] - tmp[i * n + j];
    }
    return matrix_r;
}

void sum_matrix(float* matrix_a, float* matrix_b, const int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            matrix_a[i * n + j] += matrix_b[i * n + j];
        }
    }
}

float* get_result(float* matrix_i, float* matrix_r, const int n, const int m)
{
    float* result = calloc(n * n, sizeof(float));
    result = fill_unit_matrix(n);
    float* matrix_tmp = calloc(n * n, sizeof(float));
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            matrix_tmp[i * n + j] = matrix_r[i * n + j];
    }
    for (int i = 0; i < m; i++)
    {
        sum_matrix(result, matrix_tmp, n);
        matrix_tmp = matrix_multiplication(matrix_tmp, matrix_r, n);
    }
    free(matrix_tmp);
    return result;
}

int main(void)
{
    int n;
    int m;
    if ((scanf("%d\n", &n)) != 1)
        exit(1);
    if((scanf("%d", &m)) != 1)
        exit(1);
    float start_time = clock();

    float* matrix_a = fill_matrix_a(n);
    float* matrix_b = fill_matrix_b(matrix_a, n);
    float* matrix_i = fill_unit_matrix(n);
    float* matrix_r = get_matrix_r(matrix_i, matrix_a, matrix_b, n);
    float* result = get_result(matrix_i, matrix_r, n, m);
    result = matrix_multiplication(result, matrix_b, n);
    free(matrix_a);
    free(matrix_b);
    free(matrix_i);
    free(matrix_r);
    free(result);

    float end_time = clock();
    float search_time = (end_time - start_time)/1000.0;

    printf("%d", &search_time);

    return 0;
}