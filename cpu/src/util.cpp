#include "util.h"
#include <iostream>
using namespace std;
namespace model_X
{
	void* mylloc(size_t size, int alignment)
	{
		void* ps = (void*)malloc(sizeof(void*) + size + alignment);
		if (ps)
		{
			void* ph = (void*)((size_t)ps + sizeof(void*));
			void* palig = (void*)(((size_t)ph | (alignment - 1)) + 1);
			*((void**)palig - 1) = ps;
			return palig;
		}
		else
			return nullptr;
		//
	}
	void myfree(DTYPE*& p)
	{
		if (p)
		{
			free(*((void**)p - 1));
			p = nullptr;
		}
	}
#ifdef AVX_2
	void add_avx(DTYPE* d1, DTYPE* d2, DTYPE* des, const uint32_t& size)
	{
		if (size > 8)
		{
			__m256 data_block_temp;
			__m256 data_loader1;
			__m256 data_loader2;
			uint32_t end, k = 0;
			while (true)
			{
				end = k + 8;
				if (end >= size)
					break;
				data_loader1 = _mm256_load_ps(d1 + k);
				data_loader2 = _mm256_load_ps(d2 + k);
				data_block_temp = _mm256_add_ps(data_loader1, data_loader2);
				_mm256_storeu_ps(des + k, data_block_temp);
				k = end;
			}
			for (uint32_t i = k; i < size; i++)
				des[i] = d1[i] + d2[i];
		}
		else if (size > 0)
			add_normal(d1, d2, des, size);
	}
	DTYPE muladd_avx(DTYPE* d1, DTYPE* d2, uint32_t& size)
	{
		if (size > 8)
		{
			__m256 data_block = _mm256_setzero_ps();
			__m256 data_loader1;
			__m256 data_loader2;
			uint32_t k = 0;
			uint32_t end;
			while (true)
			{
				end = k + 8;
				if (end >= size)
					break;
				data_loader1 = _mm256_load_ps(d1 + k);
				data_loader2 = _mm256_load_ps(d2 + k);
				data_block = _mm256_fmadd_ps(data_loader1, data_loader2, data_block);
				k = end;
			}
			data_block = _mm256_hadd_ps(data_block, data_block);
			data_block = _mm256_hadd_ps(data_block, data_block);
			DTYPE out = data_block.m256_f32[0] + data_block.m256_f32[4];
			for (uint32_t i = k; i < size; i++)
				out += d1[i] * d2[i];
			return out;
		}
		else if (size > 0)
			return muladd_normal(d1, d2, size);
		else return 0;
	}
	DTYPE sum_avx(DTYPE* d1, uint32_t& size)
	{
		if (size > 8)
		{
			__m256 data_block = _mm256_setzero_ps();
			__m256 data_loader1;
			uint32_t end, k = 0;
			while (true)
			{
				end = k + 8;
				if (end >= size)
					break;
				data_loader1 = _mm256_load_ps(d1 + k);
				data_block = _mm256_add_ps(data_block, data_loader1);
				k = end;
			}
			data_block = _mm256_hadd_ps(data_block, data_block);
			data_block = _mm256_hadd_ps(data_block, data_block);
			DTYPE out = data_block.m256_f32[0] + data_block.m256_f32[4];
			for (uint32_t i = k; i < size; i++)
				out += d1[i];
			return out;
		}
		else if (size > 0)
			return sum_normal(d1, size);
		else return 0;
	}
	DTYPE var_avx(DTYPE* d1, DTYPE& mean, uint32_t& size)
	{
		DTYPE out = 0;
		if (size > 8)
		{
			__m256 data_block = _mm256_set1_ps(mean);
			__m256 data_block1 = _mm256_setzero_ps();
			__m256 data_block_temp;
			__m256 data_loader1;
			uint32_t end, k = 0;
			while (true)
			{
				end = k + 8;
				if (end >= size)
					break;
				data_loader1 = _mm256_load_ps(d1 + k);
				data_block_temp = _mm256_sub_ps(data_loader1, data_block);
				data_block1 = _mm256_fmadd_ps(data_block_temp, data_block_temp, data_block1);
				k = end;
			}
			data_block = _mm256_hadd_ps(data_block, data_block);
			data_block = _mm256_hadd_ps(data_block, data_block);
			DTYPE out = data_block.m256_f32[0] + data_block.m256_f32[4];
			for (uint32_t i = k; i < size; i++)
			{
				out += pow(d1[i] - mean, 2);
			}
			return out / size;
		}
		else if (size > 0)
			return var_normal(d1, mean, size);
		else return 0;
	}
	void linear_muladd_avx(DTYPE* d1, DTYPE& d2, DTYPE& d3, uint32_t& size)
	{
		if (size > 8)
		{
			__m256 data_block1 = _mm256_set1_ps(d2);
			__m256 data_block2 = _mm256_set1_ps(d3);
			__m256 data_block_temp;
			__m256 data_loader1;
			uint32_t end, k = 0;
			while (true)
			{
				end = k + 8;
				if (end >= size)
					break;
				data_loader1 = _mm256_load_ps(d1 + k);
				data_block_temp = _mm256_fmadd_ps(data_loader1, data_block1, data_block2);
				_mm256_storeu_ps(d1 + k, data_block_temp);
				k = end;
			}
			for (uint32_t i = k; i < size; i++)
			{
				d1[i] = d1[i] * d2 + d3;
			}
		}
		else
			linear_muladd_normal(d1, d2, d3, size);
	}
	void apply_gradients_avx(DTYPE * grad, DTYPE * data, DTYPE lr, uint32_t & size)
	{
		if (size > 8)
		{
			__m256 data_loader;
			__m256 grad_loader;
			__m256 learningrate = _mm256_set1_ps(lr);
			__m256 temp;
			uint32_t end, k = 0;
			while (true)
			{
				end = k + 8;
				if (end >= size)
					break;
				grad_loader = _mm256_loadu_ps(grad + k);
				data_loader = _mm256_loadu_ps(data + k);
				temp = _mm256_fnmadd_ps(grad_loader, learningrate, data_loader);
				_mm256_storeu_ps(data + k, temp);
				k = end;
			}
			for (uint32_t i = k; i < size; i++)
				data[i] = data[i] - lr * grad[i];
		}
		else
			apply_gradients_normal(grad, data, lr, size);
	}
#endif
	void add_normal(DTYPE* d1, DTYPE* d2, DTYPE* des, const uint32_t& size)
	{
		for (uint32_t i = 0; i < size; i++)
			des[i] = d1[i] + d2[2];
	}
	DTYPE muladd_normal(DTYPE* d1, DTYPE* d2, uint32_t& size)
	{
		DTYPE out = 0;
		for (uint32_t i = 0; i < size; i++)
			out += d1[i] * d2[i];
		return out;
	}
	DTYPE sum_normal(DTYPE* d1, uint32_t& size)
	{
		DTYPE out = 0;
		for (uint32_t i = 0; i < size; i++)
			out += d1[i];
		return out;
	}
	DTYPE var_normal(DTYPE* d1, DTYPE& mean, uint32_t& size)
	{
		DTYPE out = 0;
		for (uint32_t i = 0; i < size; i++)
			out += (d1[i] - mean) * (d1[i] - mean);
		return out / size;
	}
	void apply_gradients_normal(DTYPE * grad, DTYPE * data, DTYPE lr, uint32_t & size)
	{
		for (uint32_t i = 0; i < size; i++)
		{
			data[i] = data[i] - grad[i] * lr;
		}
	}
	void linear_muladd_normal(DTYPE* d1, DTYPE& d2, DTYPE& d3, uint32_t& size)
	{
		for (uint32_t i = 0; i < size; i++)
			d1[i] = d1[i] * d2 + d3;
	}
	int get_cpu_cors()
	{
#ifdef _WIN32
		SYSTEM_INFO sysInfo;
		GetSystemInfo(&sysInfo);
		return sysInfo.dwNumberOfProcessors;
#else
		return (int)sysconf(_SC_NPROCESSORS_ONLN);
#endif
	}
}