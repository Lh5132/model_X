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
	storage Operator::forward(storage input) { return storage(); }
	void Operator::pass_gradients()
	{
		if (pre)
		{
			if (pre->count_out <= 1)
				pre->dL_dout = dL_din;
			else
			{
				if (pre->dL_dout)
					ADD(pre->dL_dout->data, dL_din->data, pre->dL_dout->data, dL_din->total_size);
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
	Conv_2d::layer_op() :
		transform_matrix(nullptr),
		k_w(0),
		k_h(0),
		in_channels(0),
		out_channels(0),
		kernel_size(0),
		kernel_steps(0),
		total_size(0),
		weights(nullptr),
		bias(nullptr),
		strid(conv_stride()),
		padding(conv_padding()),
		with_bias(false)
	{
		ID = LAYER_ID::CONV_2D;
	}

	Conv_2d::layer_op(
		uint16_t in_channels, 
		uint16_t out_channels, 
		uint8_t w, uint8_t h, 
		conv_stride strid, 
		conv_padding padding,
		bool with_bias)
		: transform_matrix(nullptr),
		k_w(w),
		k_h(h),
		in_channels(in_channels),
		out_channels(out_channels),
		kernel_size(w*h),
		strid(strid),
		padding(padding),
		with_bias(with_bias)
	{
		ID = LAYER_ID::CONV_2D;
		kernel_steps = kernel_size*in_channels;
		uint8_t tail = kernel_steps % DATA_ALIGN;
		if (tail == 0)
		{
			data_pad = 0;
			kernel_steps_pad = kernel_steps;
		}
		else
		{
			data_pad = DATA_ALIGN - tail;
			kernel_steps_pad = kernel_steps + data_pad;
		}

		total_size = kernel_steps_pad * out_channels;
		weights = (float*)mylloc(total_size * DBYTES, MALLOC_ALIGN);

		if (kernel_steps_pad > kernel_steps)
		{
			for (int c = 0; c < out_channels; c++)
			{
				for (uint32_t p = kernel_steps; p < kernel_steps_pad; p++)
					get_channel_data(c)[p] = 0;
			}
		}
		if(with_bias)
			bias = new DTYPE[out_channels]{};
		else bias = nullptr;
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

	void __conv_async_helper(storage input, storage out, Conv_2d* conv,
		uint32_t tm_batch_steps, uint32_t out_cols,
		uint32_t start, uint32_t end)
	{
		for (uint32_t ii = 0; ii < end; ii++)
		{
			int out_row_loc = ii / out_cols;
			int out_col_loc = ii - out_row_loc*out_cols;
			int i = out_row_loc*conv->strid.h - conv->padding.top;
			int j = out_col_loc*conv->strid.w - conv->padding.left;
			uint32_t tm_row_loc = ii * conv->tm_cols_pad;
			DTYPE* dout_dw_channel;
			uint32_t batch_loc, dw_where;
			for (uint16_t b = 0; b < input->batchsize; b++)
			{
				uint32_t batch_start = b*tm_batch_steps;
				if (conv->is_gradiets())
				{
					batch_loc = b*conv->tm_cols*conv->tm_rows;
				}
				for (uint16_t c = 0; c < input->channels; c++)
				{
					uint32_t tm_row_ch_loc = c*conv->kernel_size;
					DTYPE* input_data = input->get_channel_data(b, c);
					for (uint8_t krow = 0; krow < conv->k_h; krow++)
					{
						for (uint8_t kcol = 0; kcol < conv->k_w; kcol++)
						{
							uint32_t where = batch_start + tm_row_loc + tm_row_ch_loc + krow*conv->k_w + kcol;
							if (conv->is_gradiets())
								dw_where = batch_loc + (tm_row_ch_loc + krow*conv->k_w + kcol)*conv->tm_rows + ii;
							if ((i + krow) >= 0 && (i + krow) < input->rows && (j + kcol) >= 0 && (j + kcol) < input->cols)
							{
								conv->transform_matrix[where] = *(input_data + (i + krow)*input->cols + j + kcol);
								if (conv->is_gradiets())
								{
									conv->dout_dw[dw_where] = conv->transform_matrix[where];
									if (c == 0)
									{
										uint32_t inp_loc = (i + krow)*input->cols + (j + kcol);
										conv->dout_din_w_loc[inp_loc*conv->kernel_size + conv->dout_din_row_size[inp_loc]] = krow*conv->k_w + kcol;
										conv->dout_din_out_loc[inp_loc*conv->kernel_size + conv->dout_din_row_size[inp_loc]] = ii;
										conv->dout_din_row_size[inp_loc] += 1;
									}
								}
							}
							else
							{
								conv->transform_matrix[where] = 0;
								if (conv->require_gradients)
									conv->dout_dw[dw_where] = 0;
							}
						}
					}
				}
				if (conv->data_pad > 0)
				{
					uint32_t pad_start = batch_start + tm_row_loc + input->channels * conv->kernel_size;
					for (uint8_t t = 0; t < conv->data_pad; t++)
						conv->transform_matrix[pad_start + t] = 0;
				}
				for (uint16_t cc = 0; cc < out->channels; cc++)
				{
					if (conv->with_bias)
						out->get_channel_data(b, cc)[ii] = MUL_ADD(conv->get_channel_data(cc),
							conv->transform_matrix + batch_start + tm_row_loc, conv->tm_cols_pad) + conv->bias[cc];
					else
						out->get_channel_data(b, cc)[ii] = MUL_ADD(conv->get_channel_data(cc),
							conv->transform_matrix + batch_start + tm_row_loc, conv->tm_cols_pad);
				}
			}
		}
	}


	storage Conv_2d::forward(storage input)
	{
		if (input->channels != in_channels)
			throw "number of channels mismatch!";
		if (padding.Padding_style == PADDING_STYLE::SAME)
		{
			uint8_t pad_w = (input->cols - 1)*strid.h - input->cols + k_w;
			uint8_t pad_h = (input->rows - 1)*strid.w - input->rows + k_h;
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
		uint32_t out_rows = (input->rows + padding.top + padding.bottom - k_h) / strid.h + 1;
		uint32_t out_cols = (input->cols + padding.left + padding.right - k_w) / strid.w + 1;
		storage out = storage_creater::creat(input->batchsize, out_channels, out_rows, out_cols);
		//构建中间矩阵
		tm_cols = kernel_steps;
		tm_cols_pad = kernel_steps_pad;
		tm_rows = out_rows*out_cols;
		uint32_t tm_batch_steps = tm_rows*tm_cols_pad;
		uint32_t tm_size = tm_batch_steps*input->batchsize;
		transform_matrix = (DTYPE*)mylloc(tm_size*DBYTES, MALLOC_ALIGN);
		if (require_gradients)
		{
			if(!dout_dw)
				dout_dw = (DTYPE*)mylloc(tm_cols*tm_rows*input->batchsize*DBYTES, DATA_ALIGN);
			if (!dL_din)
				dL_din = input->copy(false);
			if(!dout_din_w_loc)
				dout_din_w_loc = new uint16_t[input->channel_steps*kernel_size]{};
			if (!dout_din_out_loc)
				dout_din_out_loc = new uint16_t[input->channel_steps*kernel_size]{};
			if(!dout_din_row_size)
				dout_din_row_size = new uint16_t[input->channel_steps]{};
			out->require_grad();
		}
		if (parall_thread > 1)
		{
			uint32_t base_n = out->channel_steps / parall_thread;
			future<void>* fn = new future<void>[parall_thread];
			for (uint8_t i = 0; i < parall_thread - 1; i++)
			{
				fn[i] = async(launch::async, __conv_async_helper, input, out, this,
					tm_batch_steps, out_cols, base_n*i, base_n*(i + 1));
			}
			fn[parall_thread - 1] = async(launch::async, __conv_async_helper, input, out, this,
				tm_batch_steps, out_cols, base_n*(parall_thread - 1), out->channel_steps);
			for (int i = 0; i < parall_thread; i++)
				fn[i].wait();
			delete[] fn;
		}
		else
		{
			for (uint32_t ii = 0; ii < out->channel_steps; ii++)
			{
				int out_row_loc = ii / out_cols;
				int out_col_loc = ii - out_row_loc*out_cols;
				int i = out_row_loc*strid.h - padding.top;
				int j = out_col_loc*strid.w - padding.left;
				uint32_t tm_row_loc = ii * tm_cols_pad;
				DTYPE* dout_dw_channel;
				uint32_t batch_loc, dw_where;
				for (uint16_t b = 0; b < input->batchsize; b++)
				{
					uint32_t batch_start = b*tm_batch_steps;
					if (is_gradiets())
					{
						batch_loc = b*tm_cols*tm_rows;
					}
					for (uint16_t c = 0; c < input->channels; c++)
					{
						uint32_t tm_row_ch_loc = c*kernel_size;
						DTYPE* input_data = input->get_channel_data(b, c);
						for (uint8_t krow = 0; krow < k_h; krow++)
						{
							for (uint8_t kcol = 0; kcol < k_w; kcol++)
							{
								uint32_t where = batch_start + tm_row_loc + tm_row_ch_loc + krow*k_w + kcol;
								if (is_gradiets())
									dw_where = batch_loc + (tm_row_ch_loc + krow*k_w + kcol)*tm_rows + ii;
								if ((i + krow) >= 0 && (i + krow) < input->rows && (j + kcol) >= 0 && (j + kcol) < input->cols)
								{
									transform_matrix[where] = *(input_data + (i + krow)*input->cols + j + kcol);
									if (is_gradiets())
									{
										dout_dw[dw_where] = transform_matrix[where];
										if (c == 0)
										{
											uint32_t inp_loc = (i + krow)*input->cols + (j + kcol);
											dout_din_w_loc[inp_loc*kernel_size + dout_din_row_size[inp_loc]] = krow*k_w + kcol;
											dout_din_out_loc[inp_loc*kernel_size + dout_din_row_size[inp_loc]] = ii;
											dout_din_row_size[inp_loc] += 1;
										}
									}
								}
								else
								{
									transform_matrix[where] = 0;
									if (require_gradients)
										dout_dw[dw_where] = 0;
								}
							}
						}
					}
					if (data_pad > 0)
					{
						uint32_t pad_start = batch_start + tm_row_loc + input->channels * kernel_size;
						for (uint8_t t = 0; t < data_pad; t++)
							transform_matrix[pad_start + t] = 0;
					}
					for (uint16_t cc = 0; cc < out->channels; cc++)
					{
						if (with_bias)
							out->get_channel_data(b, cc)[ii] = MUL_ADD(get_channel_data(cc),
								transform_matrix + batch_start + tm_row_loc, tm_cols_pad) + bias[cc];
						else
							out->get_channel_data(b, cc)[ii] = MUL_ADD(get_channel_data(cc),
								transform_matrix + batch_start + tm_row_loc, tm_cols_pad);
					}
				}
			}
		}
		myfree(transform_matrix);
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
		srand(clock());
		if (init_method == Uniform)
		{
			for (uint16_t c = 0; c < out_channels; c++)
			{
				for (uint32_t i = 0; i < kernel_steps; i++)
				{
					get_channel_data(c)[i] = random_uniform();
				}
			}
		}
		else if (init_method == Normal)
		{
			for (uint16_t c = 0; c < out_channels; c++)
			{
				for (uint32_t i = 0; i < kernel_steps; i++)
				{
					get_channel_data(c)[i] = random_gaussrand(0, 1);
				}
				if (kernel_steps_pad > kernel_steps)
				{
					for (uint32_t p = kernel_steps; p < kernel_steps_pad; p++)
						get_channel_data(c)[p] = 0;
				}
			}
		}
		else
			throw "Please identify init method(Normal or Uniform)";
	}
	void Conv_2d::zero_grad()
	{
		clear_concators(this);
		fill(dL_din->data, dL_din->data + dL_din->total_size, 0.0f);
		fill(dL_dout->data, dL_dout->data + dL_dout->total_size, 0.0f);
	}

	void Conv_2d::backward(Optimizer::base_optimizer& opt)
	{
		//计算dL_dw
		uint32_t dw_size = kernel_steps*out_channels;
		if(!dL_dw_now)
			dL_dw_now = new DTYPE[dw_size]{};
		if (with_bias)
			if(!dL_db_now)
				dL_db_now = new DTYPE[out_channels]{};
		for (uint16_t b = 0; b < dL_dout->batchsize; b++)
		{
			DTYPE* dout_dw_batch = dout_dw + b * tm_cols*tm_rows;
			for (uint16_t c = 0; c < out_channels; c++)
			{
				DTYPE* channel_w = get_channel_data(c);
				DTYPE* dL_dw_channel_data = dL_dw_now + c*kernel_steps;
				DTYPE* dL_dout_channle_data = dL_dout->get_channel_data(b, c);
				for (uint32_t i = 0; i < kernel_steps; i++)
				{
					dL_dw_channel_data[i] += MUL_ADD(dout_dw_batch+i*tm_rows, dL_dout_channle_data, tm_rows);
				}
				if (with_bias)
				{
					dL_db_now[c] += SUM(dL_dout_channle_data, dL_dout->channel_steps);
				}
				for (uint16_t i = 0; i < dL_din->channels; i++)
				{
					DTYPE* dl_din_channel = dL_din->get_channel_data(b,i);
					for (uint32_t j = 0; j < dL_din->channel_steps; j++)
					{
						DTYPE temp = 0;
						uint32_t loc = j*kernel_size;
						for (uint16_t k = 0; k < dout_din_row_size[j]; k++)
						{
							temp += channel_w[i*kernel_size+dout_din_w_loc[loc+k]] * \
								dL_dout_channle_data[dout_din_out_loc[loc+k]];
						}
						dl_din_channel[j] += temp;
					}
				}
			}
		}
		//for (uint16_t i = 0; i < dw_size; i++)
		//{
		//	cout << dL_dw_now[i] << ",";
		//}
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
					cout << get_channel_data(i)[j] << ",";
				}
				cout << endl;
			}
		}
	}
	void Conv_2d::print_bias()
	{
		if (bias)
		{
			for (uint16_t i = 0; i < out_channels; i++)
				cout << bias[i] << ",";
		}
		cout << endl;
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
		uint8_t tail = kernel_steps % DATA_ALIGN;
		if (tail == 0)
		{
			data_pad = 0;
			kernel_steps_pad = kernel_steps;
		}
		else
		{
			data_pad = DATA_ALIGN - tail;
			kernel_steps_pad = kernel_steps + data_pad;
		}
		total_size = kernel_steps_pad * out_channels;
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

	Conv_2d::~layer_op()
	{
		k_w = 0;
		k_h = 0;
		in_channels = 0;
		out_channels = 0;
		total_size = 0;
		kernel_size = 0;
		kernel_steps = 0;
		kernel_steps_pad = 0;
		data_pad = 0;
		tm_rows = 0;
		tm_cols = 0;
		remove_mylloc(weights);
		remove_mylloc(transform_matrix);
		remove_new(bias);
		remove_new(dL_dw);
		remove_new(dL_dw_1);
		remove_new(dL_dw_2);
		remove_new(dL_db);
		remove_new(dL_db_1);
		remove_new(dL_db_2);
		remove_storage(dL_din);
		remove_storage(dL_dout);
		remove_mylloc(dout_dw);
		remove_new(dout_din_out_loc);
		remove_new(dout_din_row_size);
		remove_new(dout_din_w_loc);
	}

	Dense::Dense():
		weights(nullptr),
		bias(nullptr),
		in_size(0),
		out_size(0),
		with_bias(false),
		data_pad(0),
		in_size_pad(0)
	{
		ID = LAYER_ID::DENSE;
	}
	Dense::Dense(uint32_t in_size, uint32_t out_size, bool with_bias):
		in_size(in_size),
		out_size(out_size),
		with_bias(with_bias)
	{
		ID = LAYER_ID::DENSE;
		uint8_t tail = in_size%DATA_ALIGN;
		if (tail == 0) data_pad = 0;
		else data_pad = DATA_ALIGN - tail;
		in_size_pad = in_size + data_pad;
		total_size = in_size_pad*out_size;
		weights = (DTYPE*)mylloc(total_size*DBYTES, MALLOC_ALIGN);
		if(with_bias)
			bias = new DTYPE[out_size]{};
		else bias = nullptr;
	}
	void Dense::random_init(int init_method)
	{
		if (init_method == Uniform)
		{
			srand(clock());
			for (uint32_t i = 0; i < out_size; i++)
			{
				for (uint32_t j = 0; j < in_size; j++)
				{
					get_channel_data(i)[j] = random_uniform();
				}
				if (data_pad > 0)
				{
					for (uint8_t j = 0; j < data_pad; j++)
						get_channel_data(i)[in_size + j] = 0;
				}
			}
		}
		else if (init_method == Normal)
		{
			for (uint32_t i = 0; i < out_size; i++)
			{
				for (uint32_t j = 0; j < in_size; j++)
				{
					get_channel_data(i)[j] = random_gaussrand(0,1);
				}
				if (data_pad > 0)
				{
					for (uint8_t j = 0; j < data_pad; j++)
						get_channel_data(i)[in_size + j] = 0;
				}
			}
		}
		else
			throw "Please identify init method(Normal or Uniform)";
	}

	void __dense_async_helper(storage input, storage out,Dense* dense, DTYPE* res, DTYPE* inp, uint32_t start, uint32_t end)
	{
		if(dense->with_bias)
			for (uint32_t j = start; j < end; j++)
				res[j] = MUL_ADD(dense->get_channel_data(j), inp, dense->in_size_pad) + dense->bias[j];
		else
			for (uint32_t j = start; j < end; j++)
				res[j] = MUL_ADD(dense->get_channel_data(j), inp, dense->in_size_pad) + dense->bias[j];
	}

	storage Dense::forward(storage input)
	{
		if (in_size_pad != input->batch_steps_pad)
			throw "dims miss match!";
		storage out = storage_creater::creat(input->batchsize, 1, 1, out_size);
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
			for (uint16_t b = 0; b < input->batchsize; b++)
			{
				DTYPE* res = out->get_batch_data(b);
				DTYPE* inp = input->get_batch_data(b);
				for (uint8_t i = 0; i < parall_thread - 1; i++)
				{
					fn[i] = async(launch::async, __dense_async_helper, input, out, this,
						res, inp, base_n*i, base_n*(i + 1));
				}
				fn[parall_thread - 1] = async(launch::async, __dense_async_helper, input, out, this,
					res, inp, base_n*(parall_thread - 1), out_size);
				for (int i = 0; i < parall_thread; i++)
					fn[i].wait();
			}
			delete[] fn;
		}
		else
		{
			if (with_bias)
			{
				for (uint16_t i = 0; i < input->batchsize; i++)
				{
					DTYPE* res = out->get_batch_data(i);
					DTYPE* inp = input->get_batch_data(i);
					for (uint32_t j = 0; j < out_size; j++)
						res[j] = MUL_ADD(get_channel_data(j), inp, in_size_pad) + bias[j];
				}
			}
			else
			{
				for (uint16_t i = 0; i < input->batchsize; i++)
				{
					DTYPE* res = out->get_batch_data(i);
					DTYPE* inp = input->get_batch_data(i);
					for (uint32_t j = 0; j < out_size; j++)
						res[j] = MUL_ADD(get_channel_data(j), inp, in_size_pad);
				}
			}
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
		for (uint16_t b = 0; b < dL_dout->batchsize; b++)
		{
			DTYPE* dL_dout_batch = dL_dout->get_batch_data(b);
			DTYPE* dL_din_batch = dL_din->get_batch_data(b);
			DTYPE* dout_dw_batch = dout_dw->get_batch_data(b);
			for (uint32_t i = 0; i < out_size; i++)
			{
				if (with_bias)
					dL_db_now[i] += dL_dout_batch[i];
				DTYPE* dL_dw_data = dL_dw_now + i*in_size;
				DTYPE* dout_din_data = weights + i*in_size_pad;
				for (uint32_t j = 0; j < in_size; j++)
				{
					dL_din_batch[j] += dL_dout_batch[i] * dout_din_data[j];
					dL_dw_data[j] += dL_dout_batch[i] * dout_dw_batch[j];
				}
			}
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
		uint8_t tail = in_size%DATA_ALIGN;
		if (tail == 0) data_pad = 0;
		else data_pad = DATA_ALIGN - tail;
		in_size_pad = in_size + data_pad;
		total_size = in_size_pad*out_size;
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
		data_pad = 0;
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

	storage Relu::forward(storage input)
	{
		if (!require_gradients)
		{
			for (uint16_t i = 0; i < input->batchsize; i++)
			{
				DTYPE* res = input->get_batch_data(i);
				for (uint32_t j = 0; j < input->batch_steps; j++)
					if (res[j] < 0)
						res[j] = 0;
			}
			return input;
		}
		else
		{
			if (!dL_din)
				dL_din = input->copy();
			for (uint16_t i = 0; i < input->batchsize; i++)
			{
				uint32_t batch_loc = i*input->batch_steps_pad;
				DTYPE* res = input->data + batch_loc;
				DTYPE* dL_din_data = dL_din->data + batch_loc;
				for (uint32_t j = 0; j < input->batch_steps; j++)
				{
					if (res[j] < 0)
					{
						res[j] = 0;
						dL_din_data[j] = 0;
					}
					else
						dL_din_data[j] = 1;
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
		for (uint16_t i = 0; i < dL_din->batchsize; i++)
		{
			uint32_t batch_loc = i*dL_din->batch_steps_pad;
			DTYPE* dL_din_data = dL_din->data + batch_loc;
			DTYPE* dL_dout_data = dL_dout->data + batch_loc;
			for (uint32_t j = 0; j < dL_din->batch_steps; j++)
			{
				if (dL_din_data[j] > 0)
					dL_din_data[j] = dL_dout_data[j];
			}
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

	storage Sigmoid::forward(storage input)
	{
		if (!require_gradients)
		{
			for (uint16_t i = 0; i < input->batchsize; i++)
			{
				DTYPE* res = input->get_batch_data(i);
				for (uint32_t j = 0; j < input->batch_steps; j++)
					res[j] = sigmoid(res[j]);
			}
			return input;
		}
		else
		{
			if (!dL_din)
				dL_din = input->copy();
			for (uint16_t i = 0; i < input->batchsize; i++)
			{
				uint32_t batch_loc = i*input->batch_steps_pad;
				DTYPE* res = input->data + batch_loc;
				DTYPE* dL_din_data = dL_din->data + batch_loc;
				for (uint32_t j = 0; j < input->batch_steps; j++)
				{
					DTYPE exp_ = exp(0 - res[j]);
					res[j] = 1 / (1 + exp_);
					dL_din_data[j] = pow(res[j], 2)*exp_;
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

	void Sigmoid::backward(Optimizer::base_optimizer & opt)
	{
		for (uint16_t i = 0; i < dL_din->batchsize; i++)
		{
			uint32_t batch_loc = i*dL_din->batch_steps_pad;
			DTYPE* dL_dout_data = dL_dout->data + batch_loc;
			DTYPE* dL_din_data = dL_din->data + batch_loc;
			for (uint32_t j = 0; j < dL_din->batch_steps; j++)
			{
				dL_din_data[j] = dL_din_data[j] * dL_dout_data[j];
			}
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

	storage Soft_max::forward(storage input)
	{
		if(!require_gradients)
		for (int i = 0; i < input->batchsize; i++)
		{
			soft_max(input->get_batch_data(i), input->get_batch_data(i), input->batch_steps);
		}
		else
		{
			if(!dL_din)
				dL_din = input->copy();
			for (int i = 0; i < input->batchsize; i++)
			{
				uint32_t batch_loc = i*input->batch_steps_pad;
				DTYPE* res = input->data + batch_loc;
				DTYPE* dL_din_data = dL_din->data + batch_loc;
				DTYPE* out_temp = new DTYPE[input->batch_steps]{};
				DTYPE total = 0;
				for (uint32_t j = 0; j < input->batch_steps; j++)
				{
					out_temp[j] = exp(res[j]);
					total += out_temp[j];
				}
				for (uint32_t j = 0; j < input->batch_steps; j++)
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
		for (int i = 0; i < dL_din->batchsize; i++)
		{
			uint32_t batch_loc = i*dL_din->batch_steps_pad;
			DTYPE* dL_dout_data = dL_dout->data + batch_loc;
			DTYPE* dL_din_data = dL_din->data + batch_loc;
			for (uint32_t j = 0; j < dL_dout->batch_steps; j++)
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
		weight(nullptr),
		bias(nullptr),
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
		eps(eps),
		weight(nullptr),
		bias(nullptr)
	{
		ID = LAYER_ID::BN;
		cur_mean = new DTYPE[channels]{};
		cur_var = new DTYPE[channels]{};
		running_mean = new DTYPE[channels]{};
		running_var = new DTYPE[channels]{};
		if (with_weights)
		{
			weight = new DTYPE[channels]{};
			bias = new DTYPE[channels]{};
			fill(weight, weight + channels, 1.0f);
		}
	}

	storage Batch_normal_2d::forward(storage input)
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
				for (uint16_t i = 0; i < input->channels; i++)
				{
					temp_M = 0;
					temp_V = 0;
					uint32_t channel_loc = i*input->channel_steps;
					for (uint16_t j = 0; j < input->batchsize; j++)
					{
						uint32_t loc = j*input->batch_steps_pad + channel_loc;
						temp_M += SUM(input->data + loc, input->channel_steps) / input->channel_steps / input->batchsize;
					}
					for (uint16_t j = 0; j < input->batchsize; j++)
					{
						uint32_t loc = j*input->batch_steps_pad + channel_loc;
						temp_V += var_normal(input->data + loc, temp_M, input->channel_steps) / input->batchsize;
					}
					running_mean[i] = (1 - moment)*cur_mean[i] + moment*temp_M;
					running_var[i] = (1 - moment)*cur_var[i] + moment*temp_V;
					cur_mean[i] = temp_M;
					cur_var[i] = temp_V;
					DTYPE var_sqrt = sqrt(temp_V + eps);
					dout_din[i] = weight[i] / var_sqrt;
					for (uint16_t j = 0; j < input->batchsize; j++)
					{
						uint32_t loc = j*input->batch_steps_pad + channel_loc;
						DTYPE* res = input->data + loc;
						DTYPE* dout_dw_data = dout_dw->data + loc;
						for (uint32_t k = 0; k < input->channel_steps; k++)
						{
							res[k] = (res[k] - temp_M) / var_sqrt;
							dout_dw_data[k] = res[k];
						}
						LINEAR_MUL_ADD(res, weight[i], bias[i], input->channel_steps);
					}
				}
			}
			else
			{
				for (uint16_t i = 0; i < input->channels; i++)
				{
					temp_M = 0;
					temp_V = 0;
					uint32_t channel_loc = i*input->channel_steps;
					for (uint16_t j = 0; j < input->batchsize; j++)
					{
						temp_M += SUM(input->get_channel_data(j, i), input->channel_steps) / input->channel_steps / input->batchsize;
					}
					for (uint16_t j = 0; j < input->batchsize; j++)
						temp_V += var_normal(input->get_channel_data(j, i), temp_M, input->channel_steps) / input->batchsize;
					running_mean[i] = (1 - moment)*cur_mean[i] + moment*temp_M;
					running_var[i] = (1 - moment)*cur_var[i] + moment*temp_V;
					cur_mean[i] = temp_M;
					cur_var[i] = temp_V;
					DTYPE var_sqrt = sqrt(temp_V + eps);
					dout_din[i] = 1 / var_sqrt;
					for (uint16_t j = 0; j < input->batchsize; j++)
					{
						DTYPE* res = input->data + j*input->batch_steps_pad + channel_loc;
						for (uint32_t k = 0; k < input->channel_steps; k++)
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
			for (uint16_t i = 0; i < input->channels; i++)
			{
				for (uint16_t j = 0; j < input->batchsize; j++)
				{
					DTYPE* res = input->get_channel_data(j, i);
					for (uint32_t k = 0; k < input->channel_steps; k++)
					{
						res[k] = (res[k] - running_mean[i]) / sqrt(running_var[i] + eps);
					}
					if (with_weights)
						LINEAR_MUL_ADD(res, weight[i], bias[i], input->channel_steps);
				}
			}
		}
		return input;
	}

	void Batch_normal_2d::backward(Optimizer::base_optimizer& opt)
	{
		if (with_weights)
		{
			for (uint16_t c = 0; c < channels; c++)
			{
				uint32_t channel_loc = dL_din->channel_steps * c;
				for (uint16_t b = 0; b < dL_din->batchsize; b++)
				{
					uint32_t loc = b*dL_din->batch_steps_pad + channel_loc;
					DTYPE* dL_din_data = dL_din->data + loc;
					DTYPE* dL_dout_data = dL_dout->data + loc;
					for (uint32_t i = 0; i < dL_din->channel_steps; i++)
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
				uint32_t channel_loc = dL_din->channel_steps * c;
				for (uint16_t b = 0; b < dL_din->batchsize; b++)
				{
					uint32_t loc = b*dL_din->batch_steps_pad + channel_loc;
					DTYPE* dL_din_data = dL_din->data + loc;
					DTYPE* dL_dout_data = dL_dout->data + loc;
					DTYPE* dout_dw_data = dout_dw->data + loc;
					dL_db_now[c] += SUM(dL_dout_data, dL_din->channel_steps);
					for (uint32_t i = 0; i < dL_din->channel_steps; i++)
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
		outfile.write((char*)running_mean, channels*DBYTES);
		outfile.write((char*)running_var, channels*DBYTES);
		if (with_weights)
		{
			outfile.write((char*)weight, channels*DBYTES);
			outfile.write((char*)bias, channels*DBYTES);
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
			weight = new DTYPE[channels]{};
			bias = new DTYPE[channels]{};
			instream.read((char*)weight, channels*DBYTES);
			instream.read((char*)bias, channels*DBYTES);
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
		if (weight) delete[] weight;
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

	storage Max_pool::forward(storage input)
	{
		storage out = storage_creater::creat(input->batchsize, input->channels, input->rows / pool_h, input->cols / pool_w);
		if (require_gradients)
		{
			if (!dL_din)
				dL_din = input->copy(false);
			dL_din->set_zero();
			for (uint16_t b = 0; b < input->batchsize; b++)
			{
				uint32_t batch_loc = b*input->batch_steps_pad;
				for (uint16_t c = 0; c < input->channels; c++)
				{
					uint32_t channel_loc = c*dL_din->channel_steps;
					DTYPE* res = out->get_channel_data(b, c);
					DTYPE* inp = input->data + batch_loc + channel_loc;
					DTYPE* dL_din_data = dL_din->data + batch_loc + channel_loc;
					for (uint16_t i = 0, ii = 0; i < input->rows; i += pool_h, ii++)
					{
						uint16_t out_row_loc = ii*out->cols;
						uint16_t inp_row_loc = i*input->cols;
						for (uint16_t j = 0, jj = 0; j < input->cols; j += pool_w, jj++)
						{
							DTYPE max = inp[inp_row_loc + j];
							uint8_t max_loc = inp_row_loc + j;
							for (uint8_t k = 0; k < pool_h; k++)
							{
								uint16_t inp_row_loc_ = (i + k)*input->cols;
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
			for (uint16_t b = 0; b < input->batchsize; b++)
			{
				uint32_t batch_loc = b*input->batch_steps_pad;
				for (uint16_t c = 0; c < input->channels; c++)
				{
					DTYPE* res = out->get_channel_data(b, c);
					DTYPE* inp = input->data + batch_loc + c*input->channel_steps;
					for (uint16_t i = 0, ii = 0; i < input->rows; i += pool_h, ii++)
					{
						uint16_t out_row_loc = ii*out->cols;
						uint16_t inp_row_loc = i*input->cols;
						for (uint16_t j = 0, jj = 0; j < input->cols; j += pool_w, jj++)
						{
							DTYPE max = inp[inp_row_loc + j];
							for (uint8_t k = 0; k < pool_h; k++)
							{
								uint16_t inp_row_loc_ = (i + k)*input->cols;
								for (uint8_t m = 0; m < pool_w; m++)
								{
									if (inp[inp_row_loc_ + j + m] > max)
										max = inp[inp_row_loc_ + j + m];
								}
							}
							res[out_row_loc + jj] = max;
						}
					}
				}
			}
		}
		return out;
	}

	void Max_pool::backward(Optimizer::base_optimizer & opt)
	{
		for (uint16_t b = 0; b < dL_din->batchsize; b++)
		{
			uint32_t batch_loc = b*dL_din->batch_steps_pad;
			for (uint16_t c = 0; c < dL_din->channels; c++)
			{
				uint32_t channel_loc = c*dL_din->channel_steps;
				DTYPE* dL_dout_data = dL_dout->get_channel_data(b, c);
				DTYPE* dL_din_data = dL_din->data + batch_loc + channel_loc;
				for (uint16_t i = 0, ii = 0; i < dL_din->rows; i += pool_h, ii++)
				{
					uint16_t out_row_loc = ii*dL_dout->cols;
					uint16_t inp_row_loc = i*dL_din->cols;
					for (uint16_t j = 0, jj = 0; j < dL_din->cols; j += pool_w, jj++)
					{
						for (uint8_t k = 0; k < pool_h; k++)
						{
							uint16_t inp_row_loc_ = (i + k)*dL_din->cols;
							for (uint8_t m = 0; m < pool_w; m++)
							{
								if (dL_din_data[inp_row_loc_ + j + m] > 0)
									dL_din_data[inp_row_loc_ + j + m] = dL_dout_data[out_row_loc+jj];
							}
						}
					}
				}
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

	storage Ave_pool::forward(storage input)
	{
		storage out = storage_creater::creat(input->batchsize, input->channels, input->rows / pool_h, input->cols / pool_w);
		for (uint16_t b = 0; b < input->batchsize; b++)
		{
			for (uint16_t c = 0; c < input->channels; c++)
			{
				DTYPE* res = out->get_channel_data(b, c);
				DTYPE* inp = input->get_channel_data(b, c);
				for (uint16_t i = 0, ii = 0; i < input->rows; i += pool_h, ii++)
				{
					uint16_t out_row_loc = ii*out->cols;
					for (uint16_t j = 0, jj = 0; j < input->cols; j += pool_w, jj++)
					{
						DTYPE sum = 0;
						for (uint8_t k = 0; k < pool_h; k++)
						{
							uint16_t inp_row_loc = (i + k)*input->cols;
							for (uint8_t m = 0; m < pool_w; m++)
							{
								sum += inp[inp_row_loc + j + m];
							}
						}
						res[out_row_loc+jj] = sum/pool_h/pool_w;
					}
				}
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
		for (uint16_t b = 0; b < dL_din->batchsize; b++)
		{
			for (uint16_t c = 0; c < dL_din->channels; c++)
			{
				DTYPE* res = dL_dout->get_channel_data(b, c);
				DTYPE* inp = dL_din->get_channel_data(b, c);
				for (uint16_t i = 0, ii = 0; i < dL_din->rows; i += pool_h, ii++)
				{
					uint16_t out_row_loc = ii*dL_dout->cols;
					for (uint16_t j = 0, jj = 0; j < dL_din->cols; j += pool_w, jj++)
					{
						DTYPE sum = 0;
						for (uint8_t k = 0; k < pool_h; k++)
						{
							uint16_t inp_row_loc = (i + k)*dL_din->cols;
							for (uint8_t m = 0; m < pool_w; m++)
							{
								inp[inp_row_loc + j + m] = res[out_row_loc + jj] / size;
							}
						}
					}
				}
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

	storage Drop_out::forward(storage input)
	{
		if (!require_gradients)
		{
			srand(clock());
			for (uint16_t i = 0; i < input->batchsize; i++)
				for (uint32_t j = 0; j < input->batch_steps; j++)
				{
					if (random_uniform() <= rate)
						input->get_batch_data(i)[j] = 0;
				}
		}
		else
		{
			if (!dL_din)
				dL_din = input->copy(false);
			dL_din->set_zero();
			srand(clock());
			for (uint16_t i = 0; i < input->batchsize; i++)
			{
				uint32_t batch_loc = i*input->batch_steps_pad;
				DTYPE* input_data = input->data + batch_loc;
				DTYPE* dL_din_data = dL_din->data + batch_loc;
				for (uint32_t j = 0; j < input->batch_steps; j++)
				{
					if (random_uniform() <= rate)
						input_data[j] = 0;
					else
						dL_din_data[j] = 1;
				}
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
		for (uint16_t i = 0; i < dL_din->batchsize; i++)
		{
			uint32_t batch_loc = i*dL_din->batch_steps_pad;
			DTYPE* dL_din_data = dL_din->data + batch_loc;
			DTYPE* dL_dout_data = dL_dout->data + batch_loc;
			for (uint32_t j = 0; j < dL_din->batch_steps; j++)
			{
				if (dL_din_data[j] > 0)
				{
					dL_din_data[j] = dL_dout_data[j];
				}
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

	storage Concator::forward(storage input)
	{
		return storage();
	}

	Operator* Concator::get_pre()
	{
		return start;
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
}