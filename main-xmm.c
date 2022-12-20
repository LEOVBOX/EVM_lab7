#include <stdio.h>
#include <stdlib.h>
#include <xmmintrin.h>

float* fill_matrix_a(const int n)
{
	float* matrix = (float*)_mm_malloc(n * n * sizeof(float), 16);
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
			matrix[i * n + j] = (float)(rand() % n);
	}
	return matrix;
}

void transposition_matrix(float* matrix, const int n)
{
	for (int i = 0; i < n; i++)
		for (int j = i; j < n; j++)
		{
			matrix[j * n + i] += matrix[i * n + j];
			matrix[i * n + j] = matrix[j * n + i] - matrix[i * n + j];
			matrix[j * n + i] -= matrix[i * n + j];
		}
}

float* fill_identity_matrix(const int n)
{
	float* identity_matrix = (float*)_mm_malloc(n * n * sizeof(float), 16);
	for (int i = 0; i < n; i++)
	{
		identity_matrix[i * n + i] = (float)1;
	}
	return identity_matrix;
}

float max_row(float* matrix, const int n)
{
	float max = 0;
	__m128* vector_matrix = (__m128*)matrix;
	for (int i = 0; i < n; i++)
	{

		float s = 0;
		__m128 vector_sum = _mm_setzero_ps();

		for (int j = 0; j < n; j += 4)
		{
			vector_sum = _mm_add_ps(vector_sum, vector_matrix[i + j]);
		}
		__m128 res = _mm_set_ps1(1);
		__m128 tmp = _mm_movehl_ps(tmp, res);
		res = _mm_add_ps(res, tmp);
		tmp = _mm_shuffle_ps(res, res, 1);
		res = _mm_add_ss(res, tmp);
		_mm_store_ss(&s, res);

		if (s > max)
			max = s;
	}
	return max;
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
	float* tmp = (float*)_mm_malloc(n * n * sizeof(float), 16);
	float row = max_row(matrix, n);
	float column = max_column(matrix, n);
	transposition_matrix(matrix, n);
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			tmp[i * n + j] = matrix[i * n + j] / (row * column);
		}
	}
	return tmp;
}

float* matrix_multiplication(float* matrix_a, float* matrix_b, const int n)
{
	float* matrix_c = (float*)_mm_malloc(n * n * sizeof(float), 16);
	for (int i = 0; i < n; i++)
	{
		float* c_i = matrix_c + i * n;
		for (int j = 0; j < n; j += 4)
		{
			_mm_store_ps(c_i + j, _mm_setzero_ps());
		}
		for (int k = 0; k < n; k++)
		{
			float* b_k = matrix_b + k * n;
			__m128 a_i_k = _mm_set_ps1(matrix_a[i * n + k]);
			for (int j = 0; j < n; j += 4)
			{
				_mm_storeu_ps(c_i + j, _mm_add_ps(_mm_load_ps(c_i + j),
						_mm_mul_ps(a_i_k, _mm_load_ps(b_k + j))));
			}

		}
	}
	return matrix_c;
}

float* get_matrix_r(const float* identity_matrix, float* matrix_a, float* matrix_b, const
int n)
{
	float* matrix_r = (float*)_mm_malloc(n * n * sizeof(float), 16);
	float* tmp = matrix_multiplication(matrix_b, matrix_a, n);
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
			matrix_r[i * n + j] = identity_matrix[i * n + j] - tmp[i * n + j];
	}
	return matrix_r;
}

void sum_matrix(float* matrix_a, float* matrix_b, float* matrix_result, int n)
{
	__m128* mm_matrix_a = (__m128*)matrix_a;
	__m128* mm_matrix_b = (__m128*)matrix_b;
	__m128* mm_matrix_result = (__m128*)matrix_result;
	for (int i = 0; i < n * n / 4; i += 2)
	{
		__m128 a1 = mm_matrix_a[i];
		__m128 b1 = mm_matrix_b[i];
		__m128 a2 = mm_matrix_a[i + 1];
		__m128 b2 = mm_matrix_b[i + 1];
		mm_matrix_result[i] = _mm_add_ps(a1, b1);
		mm_matrix_result[i + 1] = _mm_add_ps(a2, b2);
	}
}

float* get_result(float* matrix_i, float* matrix_r, const int n, const int m)
{
	float* result = (float*)_mm_malloc(n * n * sizeof(float), 16);
	result = fill_identity_matrix(n);
	float* matrix_tmp = (float*)_mm_malloc(n * n * sizeof(float), 16);
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
			matrix_tmp[i * n + j] = matrix_r[i * n + j];
	}
	for (int i = 0; i < m; i++)
	{
		sum_matrix(result, matrix_tmp, result, n);
		matrix_tmp = matrix_multiplication(matrix_tmp, matrix_r, n);
	}
	_mm_free(matrix_tmp);
	return result;
}

int main(void)
{
	int n = 0; // Matrix size.
	int m = 0; // Count of execution.
	scanf("%d\n", &n);
	scanf("%d", &m);
	float* matrix_a = fill_matrix_a(n);
	float* matrix_b = fill_matrix_b(matrix_a, n);
	float* matrix_i = fill_identity_matrix(n);
	float* matrix_r = get_matrix_r(matrix_i, matrix_a, matrix_b, n);
	float* result = get_result(matrix_i, matrix_r, n, m);
	result = matrix_multiplication(result, matrix_b, n);
	_mm_free(matrix_a);
	_mm_free(matrix_b);
	_mm_free(matrix_i);
	_mm_free(matrix_r);
	_mm_free(result);
	return 0;
}