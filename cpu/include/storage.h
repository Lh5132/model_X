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
		inline int get_dims(const int& i) const
		{
			if (i >= ndims)
				throw "dimension out of range";
			return dims.d[i];
		}
		inline uint8_t get_ndims() const
		{
			return ndims;
		}
		inline uint32_t get_dim_steps(const int& i) const
		{
			if (i >= ndims)
				throw "dimension out of range";
			return dim_steps[i];
		}
		inline const uint32_t& get_total_size()
		{
			return total_size;
		}
		inline DTYPE* get_data() const
		{
			return data;
		}
		inline void set_zero()
		{
			fill(data, data + total_size, 0.0f);
		}
		inline void set_one()
		{
			fill(data, data + total_size, 1.0f);
		}
		inline bool is_require_gradienst()
		{
			return this->require_gradients;
		}
		inline void set_require_gradienst(bool grad)
		{
			this->require_gradients = grad;
		}
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
		tensor operator+ (tensor other);
		tensor operator* (tensor other);
	};
}
