#pragma once

#include <stdint.h>
#include <random>
#include <chrono>

#ifndef __WIN32
#include "windows.h"
#else
#include "unistd.h"
#endif
#ifdef AVX_2
#include <immintrin.h>
#define MUL_ADD(A, B, SIZE) muladd_avx(A, B, SIZE)
#define ADD(A, B, C, SIZE) add_avx(A, B, C, SIZE)
#define SUM(A, SIZE) sum_avx(A, SIZE)
#define VAR(A, E, SIZE) var_avx(A, E, SIZE)
#define LINEAR_MUL_ADD(A, B, C, SIZE) linear_muladd_avx(A, B, C, SIZE)
#define APPLY_GRADIENTS(GRAD, DATA, LR, SIZE) apply_gradients_avx(GRAD, DATA, LR, SIZE)
#else
#define MUL_ADD(A, B, SIZE) muladd_normal(A, B, SIZE)
#define ADD(A, B, C, SIZE) add_normal(A, B, C, SIZE)
#define SUM(A, SIZE) sum_normal(A, SIZE)
#define VAR(A, E, SIZE) var_normal(A, E, SIZE)
#define LINEAR_MUL_ADD(A, B, C, SIZE) linear_muladd_normal(A, B, C, SIZE)
#define APPLY_GRADIENTS(GRAD,DATA,SIZE) apply_gradients_normal(GRAD,DATA,SIZE)
#endif

namespace model_X
{

using DTYPE = float;
constexpr auto DBYTES = sizeof(float);
constexpr auto MALLOC_ALIGN = 32;
constexpr auto DATA_ALIGN = 8;


	using namespace std;
	
	enum Init_method
	{
		Uniform = 1,
		Normal = 2,
	};
	inline void random_gaussrand(DTYPE* data, uint32_t size, DTYPE E = 0, DTYPE V = 0.01)
	{
		random_device rd;
		default_random_engine eng(rd());
		normal_distribution<DTYPE> dis(E, V);
		for (uint32_t i = 0; i < size; i++)
			data[i] = dis(eng);
	};
	inline void random_uniform(DTYPE* data, uint32_t size)
	{
		random_device rd;
		srand(rd());
		for(uint32_t i =0;i<size;i++)
			data[i] = (DTYPE)rand() / RAND_MAX;
	}

	//地址对齐
	void* mylloc(size_t size, int alignment);
	void myfree(DTYPE*& p);

#ifdef AVX_2
	void add_avx(DTYPE* d1, DTYPE* d2, DTYPE* des, const uint32_t& size);
	DTYPE muladd_avx(DTYPE* d1, DTYPE* d2, uint32_t& size);
	DTYPE sum_avx(DTYPE* d1, uint32_t& size);
	DTYPE var_avx(DTYPE* d1, DTYPE& mean, uint32_t& size);
	void linear_muladd_avx(DTYPE* d1, DTYPE& d2, DTYPE& d3, uint32_t& size);
	void apply_gradients_avx(DTYPE* grad, DTYPE* data, DTYPE lr, uint32_t& size);

#endif
	void add_normal(DTYPE* d1, DTYPE* d2, DTYPE* des, const uint32_t& size);
	DTYPE muladd_normal(DTYPE* d1, DTYPE* d2, uint32_t& size);
	DTYPE sum_normal(DTYPE* d1, uint32_t& size);
	DTYPE var_normal(DTYPE* d1, DTYPE& mean, uint32_t& size);
	void linear_muladd_normal(DTYPE* d1, DTYPE& d2, DTYPE& d3, uint32_t& size);
	void apply_gradients_normal(DTYPE* grad, DTYPE* data, DTYPE lr, uint32_t& size);
	int get_cpu_cors();
}
