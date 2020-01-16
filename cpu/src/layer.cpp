#ifdef CUDA
#include "../../gpu/cuda_functions.cuh"
#endif


#include "layer.h"
#include "storage.h"
#include <algorithm>
#include <cmath>
#include <future>
#include <iostream>
#include <unordered_set>
#include <vector>

#include "optimizer.h"


namespace model_X
{
	inline DTYPE sigmoid(DTYPE input)
	{
		return 1 / (1 + exp(0 - input));
	}
	inline void check_save_input(storage* input, bool save_input)
	{
		if (!save_input)
		{
			delete input;
			input = nullptr;
		}
	}
	inline void remove_new(DTYPE*& data)
	{
		if (data)
		{
			delete[] data;
			data = nullptr;
		}
	}
	inline void remove_new(uint8_t*& data)
	{
		if (data)
		{
			delete[] data;
			data = nullptr;
		}
	}
	inline void remove_new(uint16_t*& data)
	{
		if (data)
		{
			delete[] data;
			data = nullptr;
		}
	}
	inline void remove_mylloc(DTYPE*& data)
	{
		if (data)
		{
			myfree(data);
			data = nullptr;
		}
	}
	inline void remove_storage(storage*& n)
	{
		if (n)
		{
			delete n;
			n = nullptr;
		}
	}
	void clear_concators(Operator* op)
	{
		if (op->pre->ID == LAYER_ID::CONCAT)
		{
			vector<Operator*> stack = { op->pre };
			while (!stack.empty())
			{
				Concator* now = (Concator*)stack[stack.size() - 1];
				stack.pop_back();
				if (now->get_O1()->ID == LAYER_ID::CONCAT)
					stack.push_back(now->get_O1());
				if (now->get_O2()->ID == LAYER_ID::CONCAT)
					stack.push_back(now->get_O2());
				delete now;
			}
		}
	}

	void Operator::set_async_thread(int n)
	{
		this->parall_thread = n;
	}

	void Operator::set_async_thread()
	{
		this->parall_thread = cpu_cors;
	}

	Operator* Operator::get_pre()
	{
		return pre;
	}

	void Operator::pass_gradients()
	{
		if (pre)
		{
			if (pre->count_out <= 1)
				pre->dL_dout = dL_din;
			else
			{
				if (pre->dL_dout)
					ADD(pre->dL_dout->get_data(), dL_din->get_data(), pre->dL_dout->get_data(), dL_din->get_total_size());
				else
					pre->dL_dout = dL_din;
			}
		}
	}
	void Operator::pass_gradients(storage* gradients)
	{
		pre->dL_dout = gradients;
	}
	void Operator::backward(Optimizer::base_optimizer& opt) {}
	void Operator::zero_grad() {}
	void Operator::to_binay_file(ofstream& outfile) {}
	string Operator::info() { return ""; }
	void Operator::random_init(int init_method) {}
	Operator::~Operator()
	{
		remove_storage(dL_din);
		remove_storage(dL_dout);
	}
#ifdef CUDA
	void Operator::to_cuda() {}
	void Operator::to_cpu() {}
#endif



	void soft_max(DTYPE* input, DTYPE* out, uint16_t size)
	{
		DTYPE* out_temp = new DTYPE[size]{};
		DTYPE total = 0;
		for (uint16_t i = 0; i < size; i++)
		{
			out_temp[i] = exp(input[i]);
			total += out_temp[i];
		}
		for (uint16_t i = 0; i < size; i++)
		{
			out[i] = out_temp[i] / total;
		}
		delete[] out_temp;
	}
	Conv_2d::Conv_2d() :
		k_w(0),
		k_h(0),
		in_channels(0),
		out_channels(0),
		kernel_size(0),
		kernel_steps(0),
		total_size(0),
		strid(conv_stride()),
		padding(conv_padding()),
		tm_cols(0),
		tm_rows(0),
		tm_batch_steps(0),
		with_bias(false)
	{
		ID = LAYER_ID::CONV_2D;
	}

	Conv_2d::Conv_2d(
		uint16_t in_channels, 
		uint16_t out_channels, 
		uint8_t w, uint8_t h, 
		conv_stride strid, 
		conv_padding padding,
		bool with_bias)
		:
		k_w(w),
		k_h(h),
		in_channels(in_channels),
		out_channels(out_channels),
		kernel_size(w*h),
		strid(strid),
		padding(padding),
		with_bias(with_bias),
		tm_cols(0),
		tm_rows(0),
		tm_batch_steps(0)
	{
		ID = LAYER_ID::CONV_2D;
		kernel_steps = kernel_size*in_channels;
		total_size = kernel_steps * out_channels;
		weights = (float*)mylloc(total_size * DBYTES, MALLOC_ALIGN);
		weights_size = total_size;
		if (with_bias)
		{
			bias = new DTYPE[out_channels]{};
			bias_size = out_channels;
		}
	}

#ifdef CUDA
	void Conv_2d::to_cuda()
	{
		is_cuda = true;
		conv_to_cuda(this);
	}
	void Conv_2d::to_cpu()
	{
		is_cuda = false;
		conv_to_cpu(this);
	}
#endif

	void __conv_async_helper(storage* input, storage* out, Conv_2d* conv,
		uint32_t start, uint32_t end)
	{
		DTYPE* temp_w, *temp_data = (DTYPE*)mylloc(conv->kernel_steps * DBYTES, DATA_ALIGN);
		int i = (start / out->dims.d[3]) * conv->strid.h - conv->padding.top;
		int j = (start % out->dims.d[3]) * conv->strid.w - conv->padding.left - conv->strid.w;
		uint32_t out_batch_loc, out_channel_loc;
		uint32_t input_batch_loc, input_channel_loc, input_row_loc, input_loc;
		uint32_t input_row_statr_loc, input_col_statr_loc;
		uint32_t temp_channel_loc,temp_row_loc, temp_loc;
		uint32_t temp_row_start, temp_col_start;
		uint32_t tm_batch_loc, dw_where, din_where;
		int8_t krow_start, kcol_start, krow_end, kcol_end;
		for (uint32_t ii = start; ii < end; ii++)
		{
			j += conv->strid.w;
			if (j + conv->k_w > input->dims.d[3] + conv->padding.right)
			{
				j = -conv->padding.left;
				i += conv->strid.h;
			}
			out_batch_loc = 0;
			input_batch_loc = 0;
			tm_batch_loc = ii;
			fill(temp_data, temp_data + conv->kernel_steps, 0.0f);
			krow_start = i < 0 ? -i : 0;
			kcol_start = j < 0 ? -j : 0;
			krow_end = min(input->dims.d[2] - i, conv->k_h);
			kcol_end = min(input->dims.d[3] - j, conv->k_w);
			input_row_statr_loc = (i + krow_start) * input->dims.d[3];
			input_col_statr_loc = j + kcol_start;
			temp_row_start = krow_start * conv->k_w;
			temp_col_start = kcol_start;
			for (uint16_t b = 0; b < input->dims.d[0]; b++)
			{
				input_channel_loc = input_batch_loc;
				temp_channel_loc = 0;
				for (uint16_t c = 0; c < input->dims.d[1]; c++)
				{
					temp_row_loc = temp_channel_loc + temp_row_start;
					DTYPE* input_data = input->data + input_channel_loc;
					input_row_loc = input_channel_loc + input_row_statr_loc;
					for (uint8_t krow = krow_start; krow < krow_end; krow++)
					{
						temp_loc = temp_row_loc + temp_col_start;
						input_loc = input_row_loc + input_col_statr_loc;
						if (conv->require_gradients)
						{
							dw_where = tm_batch_loc + temp_loc * conv->tm_rows;
							din_where = input_loc * conv->kernel_size;
						}
						for (uint8_t kcol = kcol_start; kcol < kcol_end; kcol++)
						{
							temp_data[temp_loc] = input->data[input_loc];
							if (conv->require_gradients)
							{
								conv->dout_dw[dw_where] = input->data[input_loc];
								if (c == 0 && b == 0)
								{
									conv->dout_din_w_loc[din_where + conv->dout_din_row_size[input_loc]] = temp_loc;
									conv->dout_din_out_loc[din_where + conv->dout_din_row_size[input_loc]] = ii;
									conv->dout_din_row_size[input_loc] += 1;
									din_where += conv->kernel_size;
								}
								dw_where += conv->tm_rows;
							}
							temp_loc += 1;
							input_loc += 1;
						}
						temp_row_loc += conv->k_w;
						input_row_loc += input->dims.d[3];
					}
					input_channel_loc += input->dim_steps[1];
					temp_channel_loc += conv->kernel_size;
				}
				out_channel_loc = out_batch_loc;
				temp_w = conv->weights;
				for (uint16_t cc = 0; cc < conv->out_channels; cc++)
				{
					if (conv->with_bias)
						out->data[out_channel_loc + ii] = MUL_ADD(temp_w, temp_data, conv->tm_cols) + conv->bias[cc];
					else
						out->data[out_channel_loc + ii] = MUL_ADD(temp_w, temp_data, conv->tm_cols);
					out_channel_loc += out->dim_steps[1];
					temp_w += conv->kernel_steps;
				}
				out_batch_loc += out->dim_steps[0];
				input_batch_loc += input->dim_steps[0];
				if (conv->require_gradients)
				{
					tm_batch_loc += conv->tm_batch_steps;
				}
			}
		}
		myfree(temp_data);
	}
	tensor Conv_2d::forward(tensor& input)
	{
		if (input->dims.d[1] != in_channels && input->ndims != 4)
			throw "number of channels mismatch!";
		int out_rows, out_cols;
		if (padding.Padding_style == PADDING_STYLE::SAME)
		{
			out_rows = input->dims.d[2];
			out_cols = input->dims.d[3];
			uint8_t pad_w = (input->dims.d[3] - 1)*strid.h - input->dims.d[3] + k_w;
			uint8_t pad_h = (input->dims.d[2] - 1)*strid.w - input->dims.d[2] + k_h;
			if (pad_w % 2 == 0)
				padding.left = padding.right = pad_w / 2;
			else
			{
				padding.left = pad_w / 2 + 1;
				padding.right = pad_w / 2;
			}
			if (pad_h % 2 == 0)
				padding.top = padding.bottom = pad_w / 2;
			else
			{
				padding.top = pad_w / 2 + 1;
				padding.bottom = pad_w / 2;
			}
		}
		else
		{
			out_rows = (input->dims.d[2] + padding.top + padding.bottom - k_h) / strid.h + 1;
			out_cols = (input->dims.d[3] + padding.left + padding.right - k_w) / strid.w + 1;
		}
		tensor out({ input->get_dims(0), out_channels, out_rows, out_cols });
		//构建中间矩阵
		tm_cols = kernel_steps;
		tm_rows = out_rows*out_cols;
		tm_batch_steps = tm_rows*tm_cols;
		if (require_gradients)
		{
			uint32_t tm_size = tm_batch_steps * input->dims.d[0];
			uint32_t tm_din_size = input->dim_steps[1] * kernel_size;
			if (!dout_dw)
			{
				dout_dw = (DTYPE*)mylloc(tm_size * DBYTES, DATA_ALIGN);
				fill(dout_dw, dout_dw + tm_batch_steps, 0.0f);
			}
			if (!dL_din)
				dL_din = input->copy(false);
			if(!dout_din_w_loc)
				dout_din_w_loc = new uint16_t[tm_din_size]{};
			if (!dout_din_out_loc)
				dout_din_out_loc = new uint16_t[tm_din_size]{};
			if(!dout_din_row_size)
				dout_din_row_size = new uint16_t[input->dim_steps[1]]{};
			out->require_gradients = true;
		}
		if (parall_thread > 1)
		{
			uint32_t base_n = out->dim_steps[1] / parall_thread;
			future<void>* fn = new future<void>[parall_thread];
			for (uint8_t i = 0; i < parall_thread - 1; i++)
			{
				fn[i] = async(launch::async, __conv_async_helper, input.get(), out.get(), this,
					base_n*i, base_n*(i + 1));
			}
			fn[parall_thread - 1] = async(launch::async, __conv_async_helper, input.get(), out.get(), this,
				base_n*(parall_thread - 1), out->dim_steps[1]);
			for (int i = 0; i < parall_thread; i++)
				fn[i].wait();
			delete[] fn;
		}
		else
		{
			__conv_async_helper(input.get(), out.get(), this, 0, out->dim_steps[1]);
		}
		if (require_gradients)
		{
			if (input->creater)
			{
				pre = input->creater;
				input->creater->increase_count_out();
			}
			out->creater = this;
		}
		return out;
	}

	void Conv_2d::random_init(int init_method)
	{
		if (init_method == Uniform)
			random_uniform(weights,total_size);
		else if (init_method == Normal)
			random_gaussrand(weights,total_size);
		else
			throw "Please identify init method(Normal or Uniform)";
	}
	void Conv_2d::zero_grad()
	{
		clear_concators(this);
		if(dL_din)
			fill(dL_din->data, dL_din->data + dL_din->total_size, 0.0f);
		if(dL_dout)
			fill(dL_dout->data, dL_dout->data + dL_dout->total_size, 0.0f);
		if (dL_dw_now)
			fill(dL_dw_now, dL_dw_now + total_size, 0.0f);
		if (dL_dw_now)
			fill(dL_dw_now, dL_dw_now + total_size, 0.0f);
		if (dL_db_now)
			fill(dL_db_now, dL_db_now + out_channels, 0.0f);
	}

	void Conv_2d::backward(Optimizer::base_optimizer& opt)
	{
		//计算dL_dw
		if(!dL_dw_now)
			dL_dw_now = new DTYPE[total_size]{};
		if (with_bias)
			if(!dL_db_now)
				dL_db_now = new DTYPE[out_channels]{};
		DTYPE* dL_dout_channle_data = dL_dout->data, * dl_din_channel, * dout_dw_data, * channel_w;
		uint32_t dL_dw_now_loc, loc, loc_start, dL_din_batch_loc = 0;
		DTYPE temp;
		for (uint16_t b = 0; b < dL_dout->dims.d[0]; b++)
		{
			dL_dw_now_loc = 0;
			channel_w = weights;
			for (uint16_t c = 0; c < out_channels; c++)
			{
				dout_dw_data = dout_dw;
				for (uint32_t i = 0; i < kernel_steps; i++)
				{
					dL_dw_now[dL_dw_now_loc] += MUL_ADD(dout_dw_data, dL_dout_channle_data, tm_rows);
					dout_dw_data += tm_rows;
					dL_dw_now_loc += 1;
				}
				if (with_bias)
				{
					dL_db_now[c] += SUM(dL_dout_channle_data, dL_dout->dim_steps[1]);
				}
				dl_din_channel = dL_din->data + dL_din_batch_loc;
				loc_start = 0;
				for (uint16_t i = 0; i < dL_din->dims.d[1]; i++)
				{
					loc = 0;
					for (uint32_t j = 0; j < dL_din->dim_steps[1]; j++)
					{
						temp = 0;
						for (uint16_t k = 0; k < dout_din_row_size[j]; k++)
						{
							temp += channel_w[loc_start + dout_din_w_loc[loc + k]] * \
								dL_dout_channle_data[dout_din_out_loc[loc + k]];
						}
						loc += kernel_size;
						dl_din_channel[j] += temp;
					}
					loc_start += kernel_size;
					dl_din_channel += dL_din->dim_steps[1];
				}
				channel_w += kernel_steps;
				dL_dout_channle_data += dL_dout->dim_steps[1];
			}
			dL_din_batch_loc += dL_din->dim_steps[0];
		}
		//for (int i = 0; i < total_size; i++)
		//	cout << dL_dw_now[i] << ",";
		//cout << endl;
		//for (int i = 0; i < bias_size; i++)
		//	cout << dL_db_now[i] << ",";
		//cout << endl;
		opt.apply_gradients(this);
	}
	void Conv_2d::print_weight()
	{
		if (weights)
		{
			for (uint16_t i = 0; i < out_channels; i++)
			{
				for (uint32_t j = 0; j < kernel_steps; j++)
				{
					std::cout << get_channel_data(i)[j] << ",";
				}
				std::cout << endl;
			}
		}
	}
	void Conv_2d::print_bias()
	{
		if (bias)
		{
			for (uint16_t i = 0; i < out_channels; i++)
				std::cout << bias[i] << ",";
		}
		std::cout << endl;
	}
	void Conv_2d::to_binay_file(ofstream & outfile)
	{
		outfile.write((char*)&ID, sizeof(uint8_t));
		outfile.write((char*)&in_channels, sizeof(uint16_t));
		outfile.write((char*)&out_channels, sizeof(uint16_t));
		outfile.write((char*)&k_w, sizeof(uint8_t));
		outfile.write((char*)&k_h, sizeof(uint8_t));
		outfile.write((char*)&strid, sizeof(conv_stride));
		outfile.write((char*)&padding, sizeof(conv_padding));
		outfile.write((char*)weights, total_size * DBYTES);
		outfile.write((char*)&with_bias, sizeof(bool));
		if (with_bias)
			outfile.write((char*)bias, out_channels*DBYTES);
	}

	void Conv_2d::read_stream(ifstream& instream)
	{
		instream.read((char*)&(in_channels), sizeof(uint16_t));
		instream.read((char*)&(out_channels), sizeof(uint16_t));
		instream.read((char*)&(k_w), sizeof(uint8_t));
		instream.read((char*)&(k_h), sizeof(uint8_t));
		instream.read((char*)&(strid), sizeof(conv_stride));
		instream.read((char*)&(padding), sizeof(conv_padding));

		kernel_size = k_w*k_h;
		kernel_steps = kernel_size*in_channels;
		total_size = kernel_steps * out_channels;
		weights = (float*)mylloc(total_size * DBYTES, MALLOC_ALIGN);


		instream.read((char*)(weights), total_size * DBYTES);
		instream.read((char*)&(with_bias), sizeof(bool));
		if (with_bias)
		{
			bias = new DTYPE[out_channels]{};
			instream.read((char*)(bias), out_channels*DBYTES);
		}
	}

	string Conv_2d::info()
	{
		string out = "";
		char data[200];
		sprintf(data, "Operator Conv2d\ninput channels: %d\noutput channels: %d\nkernel size: [%d,%d]\nstrid: [%d,%d]\npadding: [%d,%d,%d,%d]\n",
			in_channels,out_channels,k_w,k_h,strid.w,strid.h,padding.left,padding.top,
			padding.right,padding.bottom);
		out += data;
		return out;
	}

	Conv_2d::~Conv_2d()
	{
		k_w = 0;
		k_h = 0;
		in_channels = 0;
		out_channels = 0;
		total_size = 0;
		kernel_size = 0;
		kernel_steps = 0;
		tm_rows = 0;
		tm_cols = 0;
		remove_mylloc(weights);
		remove_new(bias);
		remove_new(dL_dw);
		remove_new(dL_dw_1);
		remove_new(dL_dw_2);
		remove_new(dL_db);
		remove_new(dL_db_1);
		remove_new(dL_db_2);
		remove_new(dL_db_now);
		remove_new(dL_dw_now);
		remove_storage(dL_din);
		remove_storage(dL_dout);
		remove_mylloc(dout_dw);
		remove_new(dout_din_out_loc);
		remove_new(dout_din_row_size);
		remove_new(dout_din_w_loc);
	}

	Dense::Dense() :
		in_size(0),
		out_size(0),
		total_size(0),
		with_bias(false)
	{
		ID = LAYER_ID::DENSE;
	}
	Dense::Dense(uint32_t in_size, uint32_t out_size, bool with_bias):
		in_size(in_size),
		out_size(out_size),
		with_bias(with_bias),
		total_size(in_size* out_size)
	{
		ID = LAYER_ID::DENSE;
		weights = (DTYPE*)mylloc(total_size * DBYTES, MALLOC_ALIGN);
		weights_size = total_size;
		if (with_bias)
		{
			bias = new DTYPE[out_size]{};
			bias_size = out_size;
		}
	}
	void Dense::random_init(int init_method)
	{
		if (init_method == Uniform)
			random_uniform(weights,total_size);
		else if (init_method == Normal)
			random_gaussrand(weights, total_size);
		else
			throw "Please identify init method(Normal or Uniform)";
	}

	void __dense_async_helper(storage* input, storage* out, Dense* dense,
		uint32_t start, uint32_t end)
	{
		DTYPE* weights = dense->get_channel_data(start);
		DTYPE* temp_weights;
		DTYPE* res = out->data;
		DTYPE* inp = input->data;
		if (dense->with_bias)
		{
			for (uint16_t b = 0; b < input->dims.d[0]; b++)
			{
				temp_weights = weights;
				for (uint32_t j = start; j < end; j++)
				{
					res[j] = MUL_ADD(temp_weights, inp, dense->in_size) + dense->bias[j];
					temp_weights += dense->in_size;
				}
				res += out->dims.d[0];
				inp += input->dims.d[0];
			}
		}
		else
		{
			for (uint16_t b = 0; b < input->dims.d[0]; b++)
			{
				temp_weights = weights;
				for (uint32_t j = start; j < end; j++)
				{
					res[j] = MUL_ADD(temp_weights, inp, dense->in_size);
					temp_weights += dense->in_size;
				}
				res += out->dims.d[0];
				inp += input->dims.d[0];
			}
		}
	}

	tensor Dense::forward(tensor& input)
	{
		if (in_size != input->dims.d[1] || input->ndims !=2)
			throw "dims miss match!";
		dimension d = { (int)input->dims.d[0], (int)out_size };
		tensor out(d);
		if (require_gradients)
		{
			if(!dout_dw)
				dout_dw = input->copy();
			if(!dL_din)
				dL_din = input->copy(false);
		}
		if (parall_thread > 1 && out_size > parall_thread)
		{
			future<void>* fn = new future<void>[parall_thread];
			uint32_t base_n = out_size / parall_thread;
			for (uint8_t i = 0; i < parall_thread - 1; i++)
			{
				fn[i] = async(launch::async, __dense_async_helper, input.get(),
					out.get(), this, base_n * i, base_n * (i + 1));
			}
			fn[parall_thread - 1] = async(launch::async, __dense_async_helper, input.get(),
				out.get(), this, base_n*(parall_thread - 1), out_size);
			for (int i = 0; i < parall_thread; i++)
				fn[i].wait();
			delete[] fn;
		}
		else
		{
			__dense_async_helper(input.get(), out.get(), this, 0, out_size);
		}
		if (require_gradients)
		{
			if (input->creater)
			{
				pre = input->creater;
				input->creater->increase_count_out();
			}
			out->creater = this;
		}
		return out;
	}

	void Dense::backward(Optimizer::base_optimizer & opt)
	{
		if (!dL_dw_now)
			dL_dw_now = new DTYPE[in_size*out_size*DBYTES]{};
		if (!dL_db_now)
			dL_db_now = new DTYPE[out_size*DBYTES]{};
		DTYPE* dL_dout_batch = dL_dout->data;
		DTYPE* dL_din_batch = dL_din->data;
		DTYPE* dout_dw_batch = dout_dw->data;
		DTYPE* dL_dw_data, * dout_din_data;
		for (uint16_t b = 0; b < dL_dout->dims.d[0]; b++)
		{
			dL_dw_data = dL_dw_now;
			dout_din_data = weights;
			for (uint32_t i = 0; i < out_size; i++)
			{
				if (with_bias)
					dL_db_now[i] += dL_dout_batch[i];
				for (uint32_t j = 0; j < in_size; j++)
				{
					dL_din_batch[j] += dL_dout_batch[i] * dout_din_data[j];
					dL_dw_data[j] += dL_dout_batch[i] * dout_dw_batch[j];
				}
				dL_dw_data += in_size;
				dout_din_data += in_size;
			}
			dL_dout_batch += dL_dout->dim_steps[0];
			dL_din_batch += dL_din->dim_steps[0];
			dout_dw_batch += dout_dw->dim_steps[0];
		}
		opt.apply_gradients(this);
	}

	void Dense::to_binay_file(ofstream& outfile)
	{
		outfile.write((char*)&ID, sizeof(uint8_t));
		outfile.write((char*)&in_size, sizeof(uint32_t));
		outfile.write((char*)&out_size, sizeof(uint32_t));
		outfile.write((char*)&with_bias, sizeof(bool));
		outfile.write((char*)weights, total_size * DBYTES);
		if(with_bias)
			outfile.write((char*)bias, out_size * DBYTES);
	}
	void Dense::read_stream(ifstream& instream)
	{
		instream.read((char*)&(in_size), sizeof(uint32_t));
		instream.read((char*)&(out_size), sizeof(uint32_t));
		instream.read((char*)&(with_bias), sizeof(bool));
		total_size = in_size*out_size;
		weights = (DTYPE*)mylloc(total_size*DBYTES, MALLOC_ALIGN);
		instream.read((char*)(weights), total_size*DBYTES);
		if (with_bias)
		{
			bias = new DTYPE[out_size]{};
			instream.read((char*)(bias), out_size*DBYTES);
		}
		else bias = nullptr;
	}
	string Dense::info()
	{
		string out = "";
		char data[150];
		sprintf(data, "Operator:Dense\ninput channels: %d\noutput channels: %d\n",
			in_size, out_size);
		out += data;
		return out;
	}
	Dense::~Dense()
	{
		in_size = 0;
		out_size = 0;
		with_bias = false;
		time_step = 0;
		remove_mylloc(weights);
		remove_new(bias);
		remove_new(dL_dw);
		remove_new(dL_dw_1);
		remove_new(dL_dw_2);
		remove_new(dL_db);
		remove_new(dL_db_1);
		remove_new(dL_db_2);
		remove_storage(dout_dw);
		remove_new(dL_dw_now);
		remove_new(dL_db_now);
		remove_storage(dL_din);
		remove_storage(dL_dout);
	}
	Relu::Relu()
	{
		ID = LAYER_ID::RELU;
	}

	tensor Relu::forward(tensor& input)
	{
		if (!require_gradients)
		{
			for (uint32_t i = 0; i < input->total_size; i++)
			{
				if (input->data[i] < 0)
					input->data[i] = 0;
			}
			return input;
		}
		else
		{
			for (uint16_t i = 0; i < input->dims.d[0]; i++)
			{
				if (!dL_din)
					dL_din = input->copy(false);
				for (uint32_t i = 0; i < input->total_size; i++)
				{
					if (input->data[i] < 0)
					{
						input->data[i] = 0;
						dL_din->data[i] = 0;
					}
					else
						dL_din->data[i] = 1;
				}
			}
			if (input->creater)
			{
				pre = input->creater;
				input->creater->increase_count_out();
			}
			input->creater = this;
			return input;
		}
	}

	void Relu::backward(Optimizer::base_optimizer & opt)
	{
		for (uint32_t j = 0; j < dL_din->total_size; j++)
		{
			if (dL_din->data[j] > 0)
				dL_din->data[j] = dL_dout->data[j];
		}
	}

	void Relu::to_binay_file(ofstream& outfile)
	{
		outfile.write((char*)&ID, sizeof(uint8_t));
	}

	void Relu::read_stream(ifstream& instream)
	{
	}

	string Relu::info()
	{
		string out = "Operator:Relu\n";
		return out;
	}


	Sigmoid::Sigmoid()
	{
		ID = LAYER_ID::SIGMOID;
	}

	tensor Sigmoid::forward(tensor& input)
	{
		if (!require_gradients)
		{
			for (uint32_t i = 0; i < input->total_size; i++)
			{
				input->data[i] = sigmoid(input->data[i]);
			}
			return input;
		}
		else
		{
			if (!dL_din)
				dL_din = input->copy(false);
			for (uint32_t i = 0; i < input->total_size; i++)
			{
				DTYPE exp_ = exp(0 - input->data[i]);
				input->data[i] = 1 / (1 + exp_);
				dL_din->data[i] = pow(input->data[i], 2)*exp_;
			}
			if (input->creater)
			{
				pre = input->creater;
				input->creater->increase_count_out();
			}
			input->creater = this;
			return input;
		}
	}

	void Sigmoid::backward(Optimizer::base_optimizer & opt)
	{
		for (uint32_t j = 0; j < dL_din->total_size; j++)
		{
			dL_din->data[j] = dL_din->data[j] * dL_dout->data[j];
		}
	}

	void Sigmoid::to_binay_file(ofstream& outfile)
	{
		outfile.write((char*)&ID, sizeof(uint8_t));
	}

	void Sigmoid::read_stream(ifstream& instream)
	{
	}

	string Sigmoid::info()
	{
		string out = "Operator:Sigmoid\n";
		return out;
	}


	Soft_max::Soft_max()
	{
		ID = LAYER_ID::SOFT_MAX;
	}

	tensor Soft_max::forward(tensor& input)
	{
		if(!require_gradients)
		for (int i = 0; i < input->dims.d[0]; i++)
		{
			soft_max(input->data + i * input->dim_steps[0], input->data + i * input->dim_steps[0], input->dim_steps[0]);
		}
		else
		{
			if(!dL_din)
				dL_din = input->copy();
			for (int i = 0; i < input->dims.d[0]; i++)
			{
				uint32_t batch_loc = i*input->dim_steps[0];
				DTYPE* res = input->data + batch_loc;
				DTYPE* dL_din_data = dL_din->data + batch_loc;
				DTYPE* out_temp = new DTYPE[input->dim_steps[0]]{};
				DTYPE total = 0;
				for (uint32_t j = 0; j < input->dim_steps[0]; j++)
				{
					out_temp[j] = exp(res[j]);
					total += out_temp[j];
				}
				for (uint32_t j = 0; j < input->dim_steps[0]; j++)
				{
					res[j] = out_temp[j] / total;
					dL_din_data[j] = out_temp[j] * (total - out_temp[j]) / pow(total, 2);
				}
				delete[] out_temp;
			}
			if (input->creater)
			{
				pre = input->creater;
				input->creater->increase_count_out();
			}
			input->creater = this;
		}
		return input;
	}

	void Soft_max::backward(Optimizer::base_optimizer & opt)
	{
		for (int i = 0; i < dL_din->dims.d[0]; i++)
		{
			uint32_t batch_loc = i*dL_din->dim_steps[0];
			DTYPE* dL_dout_data = dL_dout->data + batch_loc;
			DTYPE* dL_din_data = dL_din->data + batch_loc;
			for (uint32_t j = 0; j < dL_dout->dim_steps[0]; j++)
			{
				dL_din_data[i] = dL_din_data[j] * dL_dout_data[j];
			}
		}
	}

	void Soft_max::to_binay_file(ofstream& outfile)
	{
		outfile.write((char*)&ID, sizeof(uint8_t));
	}
	void Soft_max::read_stream(ifstream& instream)
	{
	}
	string Soft_max::info()
	{
		string out = "Operator:Soft_max\n";
		return out;
	}
	Batch_normal_2d::Batch_normal_2d()
		:with_weights(false),
		channels(0),
		moment(0),
		eps(0),
		cur_mean(nullptr),
		cur_var(nullptr),
		running_mean(nullptr),
		running_var(nullptr) 
	{
		ID = LAYER_ID::BN;
	}
	Batch_normal_2d::Batch_normal_2d(uint16_t channels, DTYPE moment, DTYPE eps, bool with_weights)
		:with_weights(with_weights),
		channels(channels),
		moment(moment),
		eps(eps)
	{
		ID = LAYER_ID::BN;
		cur_mean = new DTYPE[channels]{};
		cur_var = new DTYPE[channels]{};
		running_mean = new DTYPE[channels]{};
		running_var = new DTYPE[channels]{};
		if (with_weights)
		{
			weights = new DTYPE[channels]{};
			bias = new DTYPE[channels]{};
			weights_size = bias_size = channels;
			fill(weights, weights + channels, 1.0f);
		}
	}

	tensor Batch_normal_2d::forward(tensor& input)
	{

		if (require_gradients)
		{
			DTYPE temp_M, temp_V;
			if (!dL_din)
				dL_din = input->copy(false);
			if (!dout_din)
				dout_din = new DTYPE[channels]{};
			if (with_weights)
			{
				if (!dout_dw)
					dout_dw = input->copy(false);
				for (uint16_t i = 0; i < input->dims.d[1]; i++)
				{
					temp_M = 0;
					temp_V = 0;
					uint32_t channel_loc = i*input->dim_steps[1];
					for (uint16_t j = 0; j < input->dims.d[0]; j++)
					{
						uint32_t loc = j*input->dim_steps[0] + channel_loc;
						temp_M += SUM(input->data + loc, input->dim_steps[1]);
					}
					temp_M = temp_M / input->dim_steps[1] / input->dims.d[0];
					for (uint16_t j = 0; j < input->dims.d[0]; j++)
					{
						uint32_t loc = j*input->dim_steps[0] + channel_loc;
						temp_V += var_normal(input->data + loc, temp_M, input->dim_steps[1]);
					}
					temp_V /= input->dims.d[0];
					running_mean[i] = (1 - moment)*cur_mean[i] + moment*temp_M;
					running_var[i] = (1 - moment)*cur_var[i] + moment*temp_V;
					cur_mean[i] = temp_M;
					cur_var[i] = temp_V;
					DTYPE var_sqrt = sqrt(temp_V + eps);
					dout_din[i] = weights[i] / var_sqrt;
					for (uint16_t j = 0; j < input->dims.d[0]; j++)
					{
						uint32_t loc = j*input->dim_steps[0] + channel_loc;
						DTYPE* res = input->data + loc;
						DTYPE* dout_dw_data = dout_dw->data + loc;
						for (uint32_t k = 0; k < input->dim_steps[1]; k++)
						{
							res[k] = (res[k] - temp_M) / var_sqrt;
							dout_dw_data[k] = res[k];
						}
						LINEAR_MUL_ADD(res, weights[i], bias[i], input->dim_steps[1]);
					}
				}
			}
			else
			{
				for (uint16_t i = 0; i < input->dims.d[1]; i++)
				{
					temp_M = 0;
					temp_V = 0;
					uint32_t channel_loc = i * input->dim_steps[1];
					for (uint16_t j = 0; j < input->dims.d[0]; j++)
					{
						uint32_t loc = j * input->dim_steps[0] + channel_loc;
						temp_M += SUM(input->data + loc, input->dim_steps[1]);
					}
					temp_M = temp_M / input->dim_steps[1] / input->dims.d[0];
					for (uint16_t j = 0; j < input->dims.d[0]; j++)
					{
						uint32_t loc = j * input->dim_steps[0] + channel_loc;
						temp_V += var_normal(input->data + loc, temp_M, input->dim_steps[1]);
					}
					temp_V /= input->dims.d[0];
					running_mean[i] = (1 - moment)*cur_mean[i] + moment*temp_M;
					running_var[i] = (1 - moment)*cur_var[i] + moment*temp_V;
					cur_mean[i] = temp_M;
					cur_var[i] = temp_V;
					DTYPE var_sqrt = sqrt(temp_V + eps);
					dout_din[i] = 1 / var_sqrt;
					for (uint16_t j = 0; j < input->dims.d[0]; j++)
					{
						DTYPE* res = input->data + j*input->dim_steps[0] + channel_loc;
						for (uint32_t k = 0; k < input->dim_steps[1]; k++)
							res[k] = (res[k] - temp_M) / var_sqrt;
					}
				}
			}
			if (input->creater)
			{
				pre = input->creater;
				input->creater->increase_count_out();
			}
			input->creater = this;
		}
		else
		{
			DTYPE* res_channel_data;
			res_channel_data = input->data;
			for (uint16_t i = 0; i < input->dims.d[1]; i++)
			{
				for (uint16_t j = 0; j < input->dims.d[0]; j++)
				{
					for (uint32_t k = 0; k < input->dim_steps[1]; k++)
					{
						res_channel_data[k] = (res_channel_data[k] - running_mean[i]) / sqrt(running_var[i] + eps);
					}
					if (with_weights)
						LINEAR_MUL_ADD(res_channel_data, weights[i], bias[i], input->dim_steps[1]);
					res_channel_data += input->dim_steps[1];
				}
			}
		}
		return input;
	}

	void Batch_normal_2d::backward(Optimizer::base_optimizer& opt)
	{
		if (!with_weights)
		{
			for (uint16_t c = 0; c < channels; c++)
			{
				uint32_t channel_loc = dL_din->dim_steps[1] * c;
				for (uint16_t b = 0; b < dL_din->dims.d[0]; b++)
				{
					uint32_t loc = b*dL_din->dim_steps[0] + channel_loc;
					DTYPE* dL_din_data = dL_din->data + loc;
					DTYPE* dL_dout_data = dL_dout->data + loc;
					for (uint32_t i = 0; i < dL_din->dim_steps[1]; i++)
					{
						dL_din_data[i] = dL_dout_data[i] * dout_din[c];
					}
				}
			}
		}
		else
		{
			if (!dL_dw_now)
				dL_dw_now = new DTYPE[channels]{};
			if(!dL_db_now)
				dL_db_now = new DTYPE[channels]{};
			for (uint16_t c = 0; c < channels; c++)
			{
				uint32_t channel_loc = dL_din->dim_steps[1] * c;
				for (uint16_t b = 0; b < dL_din->dims.d[0]; b++)
				{
					uint32_t loc = b*dL_din->dim_steps[0] + channel_loc;
					DTYPE* dL_din_data = dL_din->data + loc;
					DTYPE* dL_dout_data = dL_dout->data + loc;
					DTYPE* dout_dw_data = dout_dw->data + loc;
					dL_db_now[c] += SUM(dL_dout_data, dL_din->dim_steps[1]);
					for (uint32_t i = 0; i < dL_din->dim_steps[1]; i++)
					{
						dL_dw_now[c] += dL_dout_data[i] * dout_dw_data[i];
						dL_din_data[i] += dL_dout_data[i] * dout_din[c];
					}
				}
			}
		}
	}

	void Batch_normal_2d::to_binay_file(ofstream& outfile)
	{
		outfile.write((char*)&ID, sizeof(uint8_t));
		outfile.write((char*)&channels, sizeof(uint16_t));
		outfile.write((char*)&moment, DBYTES);
		outfile.write((char*)&eps, DBYTES);
		outfile.write((char*)&with_weights, sizeof(bool));
		outfile.write((char*)running_mean, channels * DBYTES);
		outfile.write((char*)running_var, channels * DBYTES);
		if (with_weights)
		{
			outfile.write((char*)weights, channels * DBYTES);
			outfile.write((char*)bias, channels * DBYTES);
		}

	}

	void Batch_normal_2d::read_stream(ifstream& instream)
	{
		instream.read((char*)&(channels), sizeof(uint16_t));
		instream.read((char*)&(moment), DBYTES);
		instream.read((char*)&(eps), DBYTES);
		instream.read((char*)&(with_weights), sizeof(bool));
		running_mean = new DTYPE[channels]{};
		running_var = new DTYPE[channels]{};
		instream.read((char*)running_mean, channels*DBYTES);
		instream.read((char*)running_var, channels*DBYTES);
		if (with_weights)
		{
			weights = new DTYPE[channels]{};
			bias = new DTYPE[channels]{};
			instream.read((char*)weights, channels * DBYTES);
			instream.read((char*)bias, channels * DBYTES);
		}
	}

	string Batch_normal_2d::info()
	{
		string out = "";
		char data[150];
		sprintf(data, "Operator:Batch_normal_2d\nnchannels: %d\nmoment: %f\neps: %f\n",
			channels, moment, eps);
		out += data;
		return out;
	}

	Batch_normal_2d::~Batch_normal_2d()
	{
		if (weights) delete[] weights;
		if (bias) delete[] bias;
		if (running_mean) delete[] running_mean;
		if (running_var) delete[] running_var;
		if (cur_mean) delete[] cur_mean;
		if (cur_var) delete[] cur_var;
	}

	Max_pool::Max_pool(uint8_t w, uint8_t h) :
		pool_w(w),
		pool_h(h) 
	{
		ID = LAYER_ID::MAX_POOL;
	}

	tensor Max_pool::forward(tensor& input)
	{
		dimension d = { input->dims.d[0], input->dims.d[1], input->dims.d[2] / pool_h, input->dims.d[3] / pool_w };
		tensor out(d);
		if (require_gradients)
		{
			if (!dL_din)
				dL_din = input->copy(false);
			dL_din->set_zero();
			DTYPE* res = out->data;
			DTYPE* inp = input->data;
			DTYPE* dL_din_data = dL_din->data;
			for (uint16_t b = 0; b < input->dims.d[0]; b++)
			{
				uint32_t batch_loc = b*input->dim_steps[0];
				for (uint16_t c = 0; c < input->dims.d[1]; c++)
				{
					for (uint16_t i = 0, ii = 0; i < input->dims.d[2]; i += pool_h, ii++)
					{
						uint16_t out_row_loc = ii*out->dims.d[3];
						uint16_t inp_row_loc = i*input->dims.d[3];
						for (uint16_t j = 0, jj = 0; j < input->dims.d[3]; j += pool_w, jj++)
						{
							DTYPE max = inp[inp_row_loc + j];
							uint8_t max_loc = inp_row_loc + j;
							for (uint8_t k = 0; k < pool_h; k++)
							{
								uint16_t inp_row_loc_ = (i + k)*input->dims.d[3];
								for (uint8_t m = 0; m < pool_w; m++)
								{
									if (inp[inp_row_loc_ + j + m] > max)
									{
										max = inp[inp_row_loc_ + j + m];
										max_loc = inp_row_loc_ + j + m;
									}
								}
							}
							dL_din_data[max_loc] = 1;
							res[out_row_loc + jj] = max;
						}
					}
					res += out->dim_steps[1];
					inp += input->dim_steps[1];
					dL_din_data += input->dim_steps[1];
				}
			}
			if (input->creater)
			{
				pre = input->creater;
				input->creater->increase_count_out();
			}
			out->creater = this;
		}
		else
		{
			DTYPE* res = out->data;
			DTYPE* inp = input->data;
			for (uint16_t b = 0; b < input->dims.d[0]; b++)
			{
				for (uint16_t c = 0; c < input->dims.d[1]; c++)
				{
					for (uint16_t i = 0, ii = 0; i < input->dims.d[2]; i += pool_h, ii++)
					{
						uint16_t out_row_loc = ii*out->dims.d[3];
						uint16_t inp_row_loc = i*input->dims.d[3];
						for (uint16_t j = 0, jj = 0; j < input->dims.d[3]; j += pool_w, jj++)
						{
							DTYPE max = inp[inp_row_loc + j];
							for (uint8_t k = 0; k < pool_h; k++)
							{
								uint16_t inp_row_loc_ = (i + k)*input->dims.d[3];
								for (uint8_t m = 0; m < pool_w; m++)
								{
									if (inp[inp_row_loc_ + j + m] > max)
										max = inp[inp_row_loc_ + j + m];
								}
							}
							res[out_row_loc + jj] = max;
						}
					}
					res += out->dim_steps[1];
					inp += input->dim_steps[1];
				}
			}
		}
		return out;
	}

	void Max_pool::backward(Optimizer::base_optimizer & opt)
	{
		DTYPE* dL_dout_data = dL_dout->data;
		DTYPE* dL_din_data = dL_din->data;
		for (uint16_t b = 0; b < dL_din->dims.d[0]; b++)
		{
			for (uint16_t c = 0; c < dL_din->dims.d[1]; c++)
			{
				for (uint16_t i = 0, ii = 0; i < dL_din->dims.d[2]; i += pool_h, ii++)
				{
					uint16_t out_row_loc = ii*dL_dout->dims.d[3];
					uint16_t inp_row_loc = i*dL_din->dims.d[3];
					for (uint16_t j = 0, jj = 0; j < dL_din->dims.d[3]; j += pool_w, jj++)
					{
						for (uint8_t k = 0; k < pool_h; k++)
						{
							uint16_t inp_row_loc_ = (i + k)*dL_din->dims.d[3];
							for (uint8_t m = 0; m < pool_w; m++)
							{
								if (dL_din_data[inp_row_loc_ + j + m] > 0)
									dL_din_data[inp_row_loc_ + j + m] = dL_dout_data[out_row_loc+jj];
							}
						}
					}
				}
				dL_dout_data += dL_dout->dim_steps[1];
				dL_din_data += dL_din->dim_steps[1];
			}
		}
	}
	void Max_pool::to_binay_file(ofstream& outfile)
	{
		outfile.write((char*)&ID, sizeof(uint8_t));
		outfile.write((char*)&(pool_w), sizeof(uint8_t));
		outfile.write((char*)&(pool_h), sizeof(uint8_t));
	}

	void Max_pool::read_stream(ifstream& instream)
	{
		instream.read((char*)&(pool_w), sizeof(uint8_t));
		instream.read((char*)&(pool_h), sizeof(uint8_t));
	}

	string Max_pool::info()
	{
		string out = "";
		char data[150];
		sprintf(data, "Operator:Max_pool\nPool_size: [%d,%d]\n",
			pool_w, pool_h);
		out += data;
		return out;
	}

	Ave_pool::Ave_pool(uint8_t w, uint8_t h): pool_w(w),pool_h(h)
	{
		ID = LAYER_ID::AVE_POOL;
	}

	tensor Ave_pool::forward(tensor& input)
	{
		dimension d = { input->dims.d[0], input->dims.d[1], input->dims.d[2] / pool_h, input->dims.d[3] / pool_w };
		tensor out(d);
		DTYPE* res = out->data;
		DTYPE* inp = input->data;
		for (uint16_t b = 0; b < input->dims.d[0]; b++)
		{
			for (uint16_t c = 0; c < input->dims.d[1]; c++)
			{
				for (uint16_t i = 0, ii = 0; i < input->dims.d[2]; i += pool_h, ii++)
				{
					uint16_t out_row_loc = ii*out->dims.d[3];
					for (uint16_t j = 0, jj = 0; j < input->dims.d[3]; j += pool_w, jj++)
					{
						DTYPE sum = 0;
						for (uint8_t k = 0; k < pool_h; k++)
						{
							uint16_t inp_row_loc = (i + k)*input->dims.d[3];
							for (uint8_t m = 0; m < pool_w; m++)
							{
								sum += inp[inp_row_loc + j + m];
							}
						}
						res[out_row_loc+jj] = sum/pool_h/pool_w;
					}
				}
				res += out->dim_steps[1];
				inp += input->dim_steps[1];
			}
		}
		if (require_gradients)
		{
			if (!dL_din)
				dL_din = input->copy(false);
			if (input->creater)
			{
				pre = input->creater;
				input->creater->increase_count_out();
			}
			out->creater = this;
		}
		return out;
	}

	void Ave_pool::backward(Optimizer::base_optimizer & opt)
	{
		int size = pool_h * pool_w;
		DTYPE* res = dL_dout->data;
		DTYPE* inp = dL_din->data;
		for (uint16_t b = 0; b < dL_din->dims.d[0]; b++)
		{
			for (uint16_t c = 0; c < dL_din->dims.d[1]; c++)
			{
				for (uint16_t i = 0, ii = 0; i < dL_din->dims.d[2]; i += pool_h, ii++)
				{
					uint16_t out_row_loc = ii*dL_dout->dims.d[3];
					for (uint16_t j = 0, jj = 0; j < dL_din->dims.d[3]; j += pool_w, jj++)
					{
						DTYPE sum = 0;
						for (uint8_t k = 0; k < pool_h; k++)
						{
							uint16_t inp_row_loc = (i + k)*dL_din->dims.d[3];
							for (uint8_t m = 0; m < pool_w; m++)
							{
								inp[inp_row_loc + j + m] = res[out_row_loc + jj] / size;
							}
						}
					}
				}
				res += dL_dout->dim_steps[1];
				inp += dL_din->dim_steps[1];
			}
		}
	}

	void Ave_pool::to_binay_file(ofstream& outfile)
	{
		outfile.write((char*)&ID, sizeof(uint8_t));
		outfile.write((char*)&(pool_w), sizeof(uint8_t));
		outfile.write((char*)&(pool_h), sizeof(uint8_t));
	}


	void Ave_pool::read_stream(ifstream& instream)
	{
		instream.read((char*)&(pool_w), sizeof(uint8_t));
		instream.read((char*)&(pool_h), sizeof(uint8_t));
	}

	string Ave_pool::info()
	{
		string out = "";
		char data[150];
		sprintf(data, "Operator:Ave_pool\nPool_size: [%d,%d]\n",
			pool_w, pool_h);
		out += data;
		return out;
	}

	Drop_out::Drop_out(float rate):rate(rate) 
	{
		ID = LAYER_ID::DROP_OUT;
	}

	tensor Drop_out::forward(tensor& input)
	{
		srand(clock());
		if (!require_gradients)
		{
			for (uint32_t i = 0; i < input->total_size; i++)
			{
				if ((DTYPE)rand() / RAND_MAX <= rate)
					input->data[i] = 0;
			}
		}
		else
		{
			if (!dL_din)
				dL_din = input->copy(false);
			dL_din->set_zero();
			for (uint32_t i = 0; i < input->total_size; i++)
			{
				if ((DTYPE)rand() / RAND_MAX <= rate)
				{
					input->data[i] = 0;
					dL_din->data[i] = 0;
				}
				else
					dL_din->data[i] = 1;
			}
			if (input->creater)
			{
				pre = input->creater;
				input->creater->increase_count_out();
			}
			input->creater = this;
		}
		return input;
	}

	void Drop_out::backward(Optimizer::base_optimizer & opt)
	{
		for (uint32_t j = 0; j < dL_din->total_size; j++)
		{
			if (dL_din->data[j] > 0)
			{
				dL_din->data[j] = dL_dout->data[j];
			}
		}
	}

	void Drop_out::to_binay_file(ofstream& outfile)
	{
		outfile.write((char*)&ID, sizeof(uint8_t));
		outfile.write((char*)&rate, DBYTES);
	}

	void Drop_out::read_stream(ifstream& instream)
	{
		instream.read((char*)&rate, DBYTES);
	}

	string Drop_out::info()
	{
		string out = "";
		char data[150];
		sprintf(data, "Operator:Drop_out\nDrop_rate: %f\n",
			rate);
		out += data;
		return out;
	}
	Operator*& Concator::get_O1()
	{
		return O1;
	}
	Operator*& Concator::get_O2()
	{
		return O2;
	}

	tensor Concator::forward(tensor& input)
	{
		return tensor();
	}

	Add::Add(Operator* O1, Operator* O2)
	{
		ID = LAYER_ID::CONCAT;
		O1 = O1;
		O2 = O2;
		O1->increase_count_out();
		O2->increase_count_out();
	}

	void Add::set_gradients()
	{
		O1->pass_gradients(dL_dout);
		O2->pass_gradients(dL_dout);
	}
	Flaten::Flaten()
	{
	}
	tensor Flaten::forward(tensor& input)
	{
		input->flaten();
		return input;
	}
}