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
#else
#define MUL_ADD(A, B, SIZE) muladd_normal(A, B, SIZE)
#define ADD(A, B, C, SIZE) add_normal(A, B, C, SIZE)
#define SUM(A, SIZE) sum_normal(A, SIZE)
#define VAR(A, E, SIZE) var_normal(A, E, SIZE)
#define LINEAR_MUL_ADD(A, B, C, SIZE) linear_muladd_normal(A, B, C, SIZE)
#endif

namespace model_X
{

#define DTYPE float
#define DBYTES sizeof(DTYPE)
#define MALLOC_ALIGN 32
#define DATA_ALIGN 8

	using namespace std;
	
	enum Init_method
	{
		Uniform = 1,
		Normal = 2,
	};
	inline DTYPE random_gaussrand(float E, float V)
	{
		unsigned seed = chrono::system_clock::now().time_since_epoch().count();
		default_random_engine eng(seed);
		normal_distribution<DTYPE> dis(E, V);
		return dis(eng);
	};
	inline DTYPE random_uniform()
	{
		return (DTYPE)rand() / RAND_MAX;
	}

	//地址对齐
	void* mylloc(size_t size, int alignment);
	void myfree(DTYPE*& p);

#ifdef AVX_2
	void add_avx(DTYPE* d1, DTYPE* d2, DTYPE* des, uint32_t& size);
	DTYPE muladd_avx(DTYPE* d1, DTYPE* d2, uint32_t& size);
	DTYPE sum_avx(DTYPE* d1, uint32_t& size);
	DTYPE var_avx(DTYPE* d1, DTYPE& mean, uint32_t& size);
	void linear_muladd_avx(DTYPE* d1, DTYPE& d2, DTYPE& d3, uint32_t& size);
#endif
	void add_normal(DTYPE* d1, DTYPE* d2, DTYPE* des, uint32_t& size);
	DTYPE muladd_normal(DTYPE* d1, DTYPE* d2, uint32_t& size);
	DTYPE sum_normal(DTYPE* d1, uint32_t& size);
	DTYPE var_normal(DTYPE* d1, DTYPE& mean, uint32_t& size);
	void linear_muladd_normal(DTYPE* d1, DTYPE& d2, DTYPE& d3, uint32_t& size);
	int get_cpu_cors();
}
