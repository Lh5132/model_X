#ifdef CUDA
#include "../../gpu/cuda_functions.cuh"
#endif

#include "storage.h"
#include "layer.h"
#include <sstream>

namespace model_X
{
	void storage::set_up()
	{
		for (int i = 0; i < 8; i++)
		{
			if (dims.d[i] > 0)
			{
				total_size *= dims.d[i];
				ndims += 1;
			}
			dim_steps[ndims - 1] = 1;
			for (int i = ndims - 2; i >= 0; --i)
				dim_steps[i] = dims.d[i + 1] * dim_steps[i + 1];
			this->data = (float*)mylloc(this->total_size * DBYTES, MALLOC_ALIGN);
		}
	}
	storage::storage():
		ndims(0),
		total_size(0),
		dim_steps(),
		dims({ 0 }),
		data(nullptr) {}
	storage::storage(const dimension& dims) :dims(dims), total_size(1), dim_steps(),ndims(0)
	{
		set_up();
	}


#ifdef CUDA
	void storage::to_cuda()
	{
		if (!this->is_cuda)
		{
			this->is_cuda = true;
			storage_to_cuda(this);
		}
	}
	void storage::to_cpu()
	{
		if (this->is_cuda)
		{
			this->is_cuda = false;
			storage_to_cpu(this);
		}
	}
#endif

	void storage::reshape(const dimension& d)
	{
		uint32_t temp_size = 1;
		int temp_dims[8];
		uint8_t temp_n = 0;
		for (int i = 0; i < 8; i++)
		{
			if (d.d[i] == 0 || d.d[i] == -1)
				break;
			temp_size *= d.d[i];
			temp_dims[i] = d.d[i];
			temp_n += 1;
		}
		if (temp_size > total_size)
			throw "wrong dimensions";
		if (temp_size == total_size)
		{
			dims = d;
			ndims = temp_n;
			dim_steps[ndims - 1] = 1;
			for (int i = ndims - 2; i >= 0; --i)
				dim_steps[i] = dims.d[i + 1] * dim_steps[i + 1];
		}
		else
		{
			if (d.d[temp_n] == 0 || total_size % temp_size > 0 || temp_n == 8)
				throw "wrong dimensions";
			dims = d;
			dims.d[temp_n] = total_size / temp_size;
			ndims = temp_n + 1;
			dim_steps[ndims - 1] = 1;
			for (int i = ndims - 2; i >= 0; --i)
				dim_steps[i] = dims.d[i + 1] * dim_steps[i + 1];
		}
	}
	void storage::flaten()
	{
		dims.d[1] = dim_steps[0];
		dim_steps[1] = 1;
		ndims = 2;
	}
	void storage::transpose(const dimension& d)
	{
		DTYPE* temp_data = (DTYPE*)mylloc(this->total_size * DBYTES,DATA_ALIGN);
		dimension temp_dims = { 0 };
		uint32_t loc = 0;
		uint32_t temp_dim_steps[8] = { 0 };
		for (int i = 0; i < ndims; i++)
			temp_dims.d[i] = dims.d[d.d[i]];
		temp_dim_steps[ndims - 1] = 1;
		for (int i = ndims - 2; i >= 0; --i)
			temp_dim_steps[i] = temp_dims.d[i + 1] * temp_dim_steps[i + 1];
		int mini_step = dim_steps[d.d[ndims - 1]];
		for (int i = 0; i < dims.d[d.d[ndims - 1]]; i++)
		{
			temp_data[loc] = data[i * mini_step];
			loc += 1;
		}
		for (int i = ndims - 2; i >= 0; i--)
		{
			for (int j = 0; j < dims.d[d.d[i]] - 1; j++)
			{
				uint32_t src_loc = dim_steps[d.d[i]] * j;;
				for (int k = 0; k < temp_dim_steps[i]; k++)
				{
					temp_data[k + loc] = data[src_loc + k * mini_step + dim_steps[d.d[i]]];
				}
				loc += temp_dim_steps[i];
			}
		}
		dims = temp_dims;
		memcpy(dim_steps, temp_dim_steps, ndims * sizeof(int));
		myfree(data);
		data = temp_data;
	}

	void  storage::random_init(int init_method)
	{
		if (init_method == Uniform)
		{
			random_uniform(data,total_size);
		}
		else if (init_method == Normal)
		{
			random_gaussrand(data, total_size);
		}
		else
			throw "Please identify init method(Normal or Uniform)";
	}

	storage * storage::copy(bool with_data)
	{
		storage* out = new storage(dims);
		out->require_gradients = this->require_gradients;
		if(with_data)
			memcpy(out->data, data, total_size * DBYTES);
		return out;
	}

	void storage::map(DTYPE(*pf)(DTYPE))
	{
		for (uint32_t j = 0; j < this->total_size; j++)
			data[j] = pf(this->data[j]);
	}

	string storage::shape_str()
	{
		string out = "[";
		for (int i = 0; i < ndims; i++)
		{
			out += to_string(dims.d[i]);
			out += ",";
		}
		out.erase(out.end() - 1);
		out += "]";
		return out;
	}
	string get_str(int data_loca, int loc, storage* data)
	{
		if (loc == data->ndims - 1)
		{
			ostringstream out;
			out << "[";
			for (int i = 0; i < data->dims.d[loc] - 1; i++)
			{
				out << data->data[data_loca + i] << ",";
			}
			out << data->data[data_loca + data->dims.d[loc] - 1];
			out << "]";
			return out.str();
		}
		ostringstream out;
		out << "[";
		string out_s;
		for (int i = 0; i < data->dims.d[loc] - 1; i++)
		{
			out << get_str(data_loca + data->dim_steps[loc] * i, loc + 1, data);
			out << ",\n";
		}
		out << get_str(data_loca + data->dim_steps[loc] * (data->dims.d[loc] - 1), loc + 1, data);
		out << "]";
		return out.str();
	}
	string storage::data_str()
	{
		ostringstream out;
		out << "tensor\n(";
		out<< get_str(0, 0, this);
		out << ")";
		return out.str();
	}
	storage::~storage()
	{
		if(data)
			myfree(data);
	}

	tensor::tensor()
	{
	}

	tensor::tensor(const dimension& dims)
	{
		shared_ptr<storage> temp = make_shared<storage>(dims);
		temp.swap(*this);
	}

	const int* tensor::shape()
	{
		return this->get()->dims.d;
	}

	uint8_t tensor::get_ndims() const
	{
		return this->get()->ndims;
	}

	uint32_t tensor::get_dim_steps(const int& i) const
	{
		if (i >= this->get()->ndims)
			throw "dimension out of range";
		return this->get()->dim_steps[i];
	}

	const uint32_t tensor::get_total_size()
	{
		return this->get()->total_size;
	}

	DTYPE* tensor::get_data() const
	{
		return this->get()->data;
	}

	void tensor::set_zero()
	{
		fill(this->get()->data, this->get()->data + this->get()->total_size, 0.0f);
	}

	void tensor::set_one()
	{
		fill(this->get()->data, this->get()->data + this->get()->total_size, 1.0f);
	}

	bool tensor::is_require_gradienst()
	{
		return this->get()->require_gradients;
	}

	void tensor::set_require_gradienst(bool grad)
	{
		this->get()->require_gradients = grad;
	}

	tensor tensor::operator+ (tensor other)
	{
		tensor out(other);
		ADD(this->get_data(), other.get_data(), out.get_data(),other.get_total_size());
		if (this->is_require_gradienst())
		{
			if ((*this)->creater && other->creater)
			{
				Add *s = new Add((*this)->creater, other->creater);
				out->creater = s;
				(*this)->creater->increase_count_out();
				other->creater->increase_count_out();
			}
		}
		return out;
	}
}