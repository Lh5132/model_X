#pragma once

#include "util.h"
#include <memory>
#include <string>

namespace model_X
{
	typedef struct dimension
	{
		int d[8];
	} dimension;
	class Operator;
	class storage
	{
	public:
		dimension dims;
		uint8_t ndims;
		uint32_t dim_steps[8];
		uint32_t total_size;
		DTYPE* data;
		void set_up();
		bool require_gradients = false;
	public:
		Operator* creater = nullptr;    //记录每一层的信息
		storage();
		storage(const dimension& dims);
#ifdef CUDA
		storage* cuda_data; //用于管理GPU中的storage数据
		bool is_cuda;
		void to_cuda();
		void to_cpu();
#endif
		void reshape(const dimension& d);
		void flaten();
		void transpose(const dimension& d);
		void random_init(int init_method = Uniform);
		storage* copy(bool with_data = true);
		void map(DTYPE(*pf)(DTYPE));
		string shape_str();
		string data_str();
		~storage();
	};
	class tensor final : public shared_ptr<storage>
	{
	public:
		tensor();
		tensor(const dimension& dims);
		const int* shape();
		uint8_t get_ndims() const;
		uint32_t get_dim_steps(const int& i) const;
		const uint32_t get_total_size();
		DTYPE* get_data() const;
		void set_zero();
		void set_one();
		bool is_require_gradienst();
		void set_require_gradienst(bool grad);
		tensor operator+ (tensor other);
		tensor operator* (tensor other);
	};
}
