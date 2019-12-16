#ifdef CUDA
#include "../../gpu/cuda_functions.cuh"
#endif


#include "layer.h"
#include "node.h"
#include <algorithm>
#include <cmath>
#include <future>
#include <iostream>
#include <unordered_set>
#include <vector>


namespace model_X
{
	inline DTYPE sigmoid(DTYPE input)
	{
		return 1 / (1 + exp(0 - input));
	}
	inline void check_save_input(node* input, bool save_input)
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
	inline void remove_node(node*& n)
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

	Operator* Operator::get_pre()
	{
		return this->pre;
	}
	Node Operator::forward(Node input) { return Node(); }
	void Operator::pass_gradients()
	{
		if (this->pre)
		{
			if (this->pre->count_out <= 1)
				this->pre->dL_dout = this->dL_din;
			else
			{
				if (this->pre->dL_dout)
					ADD(this->pre->dL_dout->data, this->dL_din->data, this->pre->dL_dout->data, this->dL_din->total_size);
				else
					this->pre->dL_dout = this->dL_din;
			}
		}
	}
	void Operator::pass_gradients(node* gradients)
	{
		this->pre->dL_dout = gradients;
	}
	void Operator::backward(Optimizer& opt) {}
	void Operator::zero_grad() {}
	void Operator::to_binay_file(ofstream& outfile) {}
	string Operator::info() { return ""; }
	void Operator::random_init(int init_method) {}
	Operator::~Operator()
	{
		remove_node(this->dL_din);
		remove_node(this->dL_dout);
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
		this->ID = LAYER_ID::CONV_2D;
	}

	Conv_2d::Conv_2d(
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
		this->ID = LAYER_ID::CONV_2D;
		this->kernel_steps = this->kernel_size*in_channels;
		uint8_t tail = this->kernel_steps % DATA_ALIGN;
		if (tail == 0)
		{
			this->data_pad = 0;
			this->kernel_steps_pad = this->kernel_steps;
		}
		else
		{
			this->data_pad = DATA_ALIGN - tail;
			this->kernel_steps_pad = this->kernel_steps + this->data_pad;
		}

		this->total_size = this->kernel_steps_pad * out_channels;
		this->weights = (float*)mylloc(this->total_size * DBYTES, MALLOC_ALIGN);

		if (this->kernel_steps_pad > this->kernel_steps)
		{
			for (int c = 0; c < this->out_channels; c++)
			{
				for (uint32_t p = this->kernel_steps; p < this->kernel_steps_pad; p++)
					this->get_channel_data(c)[p] = 0;
			}
		}
		if(this->with_bias)
			this->bias = new DTYPE[this->out_channels]{};
		else this->bias = nullptr;
	}

#ifdef CUDA
	void Conv_2d::to_cuda()
	{
		this->is_cuda = true;
		conv_to_cuda(this);
	}
	void Conv_2d::to_cpu()
	{
		this->is_cuda = false;
		conv_to_cpu(this);
	}
#endif

	void __conv_async_helper(Node input, Node out, Conv_2d* conv,
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


	Node Conv_2d::forward(Node input)
	{
		if (input->channels != this->in_channels)
			throw "number of channels mismatch!";
		if (padding.Padding_style == PADDING_STYLE::SAME)
		{
			uint8_t pad_w = (input->cols - 1)*strid.h - input->cols + this->k_w;
			uint8_t pad_h = (input->rows - 1)*strid.w - input->rows + this->k_h;
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
		uint32_t out_rows = (input->rows + padding.top + padding.bottom - this->k_h) / strid.h + 1;
		uint32_t out_cols = (input->cols + padding.left + padding.right - this->k_w) / strid.w + 1;
		Node out = Node_creater::creat(input->batchsize, this->out_channels, out_rows, out_cols);
		//构建中间矩阵
		this->tm_cols = this->kernel_steps;
		this->tm_cols_pad = this->kernel_steps_pad;
		this->tm_rows = out_rows*out_cols;
		uint32_t tm_batch_steps = this->tm_rows*this->tm_cols_pad;
		uint32_t tm_size = tm_batch_steps*input->batchsize;
		this->transform_matrix = (DTYPE*)mylloc(tm_size*DBYTES, MALLOC_ALIGN);
		if (this->require_gradients)
		{
			if(!this->dout_dw)
				this->dout_dw = (DTYPE*)mylloc(tm_cols*tm_rows*input->batchsize*DBYTES, DATA_ALIGN);
			if (!dL_din)
				this->dL_din = input->copy(false);
			if(!dout_din_w_loc)
				this->dout_din_w_loc = new uint16_t[input->channel_steps*this->kernel_size]{};
			if (!dout_din_out_loc)
				this->dout_din_out_loc = new uint16_t[input->channel_steps*this->kernel_size]{};
			if(!dout_din_row_size)
				this->dout_din_row_size = new uint16_t[input->channel_steps]{};
			out->require_grad();
		}
		dL_din->set_zero();
		if (this->parall_thread > 1)
		{
			uint32_t base_n = out->channel_steps / this->parall_thread;
			future<void>* fn = new future<void>[this->parall_thread];
			for (uint8_t i = 0; i < this->parall_thread - 1; i++)
			{
				fn[i] = async(launch::async, __conv_async_helper, input, out, this,
					tm_batch_steps, out_cols, base_n*i, base_n*(i + 1));
			}
			fn[this->parall_thread - 1] = async(launch::async, __conv_async_helper, input, out, this,
				tm_batch_steps, out_cols, base_n*(this->parall_thread - 1), out->channel_steps);
			for (int i = 0; i < this->parall_thread; i++)
				fn[i].wait();
			delete[] fn;
		}
		else
		{
			for (uint32_t ii = 0; ii < out->channel_steps; ii++)
			{
				int out_row_loc = ii / out_cols;
				int out_col_loc = ii - out_row_loc*out_cols;
				int i = out_row_loc*this->strid.h - this->padding.top;
				int j = out_col_loc*this->strid.w - this->padding.left;
				uint32_t tm_row_loc = ii * this->tm_cols_pad;
				DTYPE* dout_dw_channel;
				uint32_t batch_loc, dw_where;
				for (uint16_t b = 0; b < input->batchsize; b++)
				{
					uint32_t batch_start = b*tm_batch_steps;
					if (this->is_gradiets())
					{
						batch_loc = b*this->tm_cols*this->tm_rows;
					}
					for (uint16_t c = 0; c < input->channels; c++)
					{
						uint32_t tm_row_ch_loc = c*this->kernel_size;
						DTYPE* input_data = input->get_channel_data(b, c);
						for (uint8_t krow = 0; krow < this->k_h; krow++)
						{
							for (uint8_t kcol = 0; kcol < this->k_w; kcol++)
							{
								uint32_t where = batch_start + tm_row_loc + tm_row_ch_loc + krow*this->k_w + kcol;
								if (this->is_gradiets())
									dw_where = batch_loc + (tm_row_ch_loc + krow*this->k_w + kcol)*tm_rows + ii;
								if ((i + krow) >= 0 && (i + krow) < input->rows && (j + kcol) >= 0 && (j + kcol) < input->cols)
								{
									this->transform_matrix[where] = *(input_data + (i + krow)*input->cols + j + kcol);
									if (this->is_gradiets())
									{
										dout_dw[dw_where] = this->transform_matrix[where];
										if (c == 0)
										{
											uint32_t inp_loc = (i + krow)*input->cols + (j + kcol);
											this->dout_din_w_loc[inp_loc*this->kernel_size + this->dout_din_row_size[inp_loc]] = krow*this->k_w + kcol;
											this->dout_din_out_loc[inp_loc*this->kernel_size + this->dout_din_row_size[inp_loc]] = ii;
											this->dout_din_row_size[inp_loc] += 1;
										}
									}
								}
								else
								{
									this->transform_matrix[where] = 0;
									if (this->require_gradients)
										dout_dw[dw_where] = 0;
								}
							}
						}
					}
					if (this->data_pad > 0)
					{
						uint32_t pad_start = batch_start + tm_row_loc + input->channels * this->kernel_size;
						for (uint8_t t = 0; t < this->data_pad; t++)
							this->transform_matrix[pad_start + t] = 0;
					}
					for (uint16_t cc = 0; cc < out->channels; cc++)
					{
						if (this->with_bias)
							out->get_channel_data(b, cc)[ii] = MUL_ADD(this->get_channel_data(cc),
								this->transform_matrix + batch_start + tm_row_loc, this->tm_cols_pad) + this->bias[cc];
						else
							out->get_channel_data(b, cc)[ii] = MUL_ADD(this->get_channel_data(cc),
								this->transform_matrix + batch_start + tm_row_loc, this->tm_cols_pad);
					}
				}
			}
		}
		myfree(this->transform_matrix);
		this->pre = input->creater;
		out->creater = this;
		return out;
	}
	void Conv_2d::random_init(int init_method)
	{
		if (init_method == Uniform)
		{
			for (uint16_t c = 0; c < this->out_channels; c++)
			{
				for (uint32_t i = 0; i < this->kernel_steps; i++)
				{
					this->get_channel_data(c)[i] = random_uniform();
				}
			}
		}
		else if (init_method == Normal)
		{
			for (uint16_t c = 0; c < this->out_channels; c++)
			{
				for (uint32_t i = 0; i < this->kernel_steps; i++)
				{
					this->get_channel_data(c)[i] = random_gaussrand(0, 1);
				}
				if (this->kernel_steps_pad > this->kernel_steps)
				{
					for (uint32_t p = this->kernel_steps; p < this->kernel_steps_pad; p++)
						this->get_channel_data(c)[p] = 0;
				}
			}
		}
		else
			throw "Please identify init method(Normal or Uniform)";
	}
	void Conv_2d::zero_grad()
	{
		clear_concators(this);
		fill(this->dL_din->data, this->dL_din->data + this->dL_din->total_size, 0.0f);
		fill(this->dL_dout->data, this->dL_dout->data + this->dL_dout->total_size, 0.0f);
	}

	void Conv_2d::backward(Optimizer& opt)
	{
		//计算dL_dw
		uint32_t dw_size = kernel_steps*out_channels;
		DTYPE* dL_dw_now = new DTYPE[dw_size]{};
		DTYPE* dL_db_now;
		if (this->with_bias)
			dL_db_now = new DTYPE[this->out_channels]{};
		for (uint16_t b = 0; b < this->dL_dout->batchsize; b++)
		{
			DTYPE* dout_dw_batch = this->dout_dw + b * tm_cols*tm_rows;
			for (uint16_t c = 0; c < this->out_channels; c++)
			{
				DTYPE* channel_w = this->get_channel_data(c);
				DTYPE* dL_dw_channel_data = dL_dw_now + c*this->kernel_steps;
				DTYPE* dL_dout_channle_data = this->dL_dout->get_channel_data(b, c);
				for (uint32_t i = 0; i < this->kernel_steps; i++)
				{
					dL_dw_channel_data[i] += MUL_ADD(dout_dw_batch+i*tm_rows, dL_dout_channle_data, this->tm_rows);
				}
				if (this->with_bias)
				{
					dL_db_now[c] += SUM(dL_dout_channle_data, this->dL_dout->channel_steps);
				}
				for (uint16_t i = 0; i < this->dL_din->channels; i++)
				{
					DTYPE* dl_din_channel = this->dL_din->get_channel_data(b,i);
					for (uint32_t j = 0; j < this->dL_din->channel_steps; j++)
					{
						DTYPE temp = 0;
						uint32_t loc = j*this->kernel_size;
						for (uint16_t k = 0; k < this->dout_din_row_size[j]; k++)
						{
							temp += channel_w[i*this->kernel_size+this->dout_din_w_loc[loc+k]] * \
								dL_dout_channle_data[this->dout_din_out_loc[loc+k]];
						}
						dl_din_channel[j] += temp;
					}
				}
			}
		}
		for (uint16_t i = 0; i < dw_size; i++)
		{
			cout << dL_dw_now[i] << ",";
		}
		cout << endl;
		//for (uint16_t i = 0; i < out_channels; i++)
		//{
		//	cout << dL_db_now[i] << ",";
		//}
		//cout << endl;
		switch (opt.optimizer_method)
		{
		case Optimizer_method::SGD:
			this->dL_dw = dL_dw_now;
			if(this->with_bias)
				this->dL_db = dL_db_now;
			break;
		case Optimizer_method::Momentum:
			if (!this->dL_dw)
			{
				this->dL_dw = new DTYPE[dw_size]{};
			}
			for (uint32_t i = 0; i < dw_size; i++)
			{
				this->dL_dw[i] = this->dL_dw[i] * opt.momentum_1 + (1 - opt.momentum_1)*dL_dw_now[i];
			}
			if (this->with_bias)
			{
				if (!this->dL_db)
					this->dL_db = new DTYPE[this->out_channels];
				for (uint16_t c = 0; c < this->out_channels;c++)
					this->dL_db[c] = this->dL_db[c] * opt.momentum_1 + (1 - opt.momentum_1)*dL_db_now[c];
			}
		case Optimizer_method::RMSProp:
			if (!this->dL_dw)
			{
				this->dL_dw = new DTYPE[dw_size]{};
			}
			if (!this->dL_dw_2)
			{
				this->dL_dw_2 = new DTYPE[dw_size]{};
			}
			for (uint32_t i = 0; i < dw_size; i++)
			{
				this->dL_dw_2[i] = this->dL_dw_2[i] * opt.momentum_2 + (1 - opt.momentum_2)*dL_dw_now[i] * dL_dw_now[i];
				this->dL_dw[i] = dL_dw_now[i] / (sqrt(this->dL_dw_2[i]) + opt.eps);
			}
			if (this->with_bias)
			{
				if (!this->dL_db)
				{
					this->dL_db = new DTYPE[this->out_channels]{};
				}
				if (!this->dL_db_2)
				{
					this->dL_db_2 = new DTYPE[this->out_channels]{};
				}
				for (uint16_t c = 0; c < this->out_channels; c++)
				{
					this->dL_db_2[c] = this->dL_db_2[c] * opt.momentum_1 + (1 - opt.momentum_1)*dL_db_now[c] * dL_db_now[c];
					this->dL_db[c] = dL_db_now[c] / (sqrt(this->dL_db_2[c]) + opt.eps);
				}
			}
		case Optimizer_method::Adam:
			this->time_step += 1;
			if (!this->dL_dw)
			{
				this->dL_dw = new DTYPE[dw_size]{};
			}
			if (!this->dL_dw_1)
			{
				this->dL_dw_1 = new DTYPE[dw_size]{};
			}
			if (!this->dL_dw_2)
			{
				this->dL_dw_2 = new DTYPE[dw_size]{};
			}

			for (uint32_t i = 0; i < dw_size; i++)
			{
				this->dL_dw_1[i] = this->dL_dw_1[i] * opt.momentum_1 + (1 - opt.momentum_1)*dL_dw_now[i];
				this->dL_dw_2[i] = this->dL_dw_2[i] * opt.momentum_2 + (1 - opt.momentum_2)*dL_dw_now[i] * dL_dw_now[i];
				this->dL_dw[i] = dL_dw_1[i] / (1 - pow(opt.momentum_1, this->time_step)) / \
					(sqrt(dL_dw_2[i] / (1 - pow(opt.momentum_2, this->time_step))) + opt.eps);
			}
			if (this->with_bias)
			{
				if (!this->dL_db)
				{
					this->dL_db = new DTYPE[this->out_channels]{};
				}
				if (!this->dL_dw_1)
				{
					this->dL_dw_1 = new DTYPE[this->out_channels]{};
				}
				if (!this->dL_dw_2)
				{
					this->dL_dw_2 = new DTYPE[this->out_channels]{};
				}

				for (uint32_t i = 0; i < this->out_channels; i++)
				{
					this->dL_db_1[i] = this->dL_db_1[i] * opt.momentum_1 + (1 - opt.momentum_1)*dL_db_now[i];
					this->dL_db_2[i] = this->dL_db_2[i] * opt.momentum_2 + (1 - opt.momentum_2)*dL_db_now[i] * dL_db_now[i];
					this->dL_db[i] = dL_db_1[i] / (1 - pow(opt.momentum_1, this->time_step)) / \
						(sqrt(dL_db_2[i] / (1 - pow(opt.momentum_2, this->time_step))) + opt.eps);
				}
			}
		default:
			break;
		}

		delete[] dL_db_now;
		delete[] dL_dw_now;
		if (opt.optimizer_method == Optimizer_method::SGD)
		{
			dL_db = nullptr;
			dL_dw = nullptr;
		}

	}
	void Conv_2d::print_weight()
	{
		if (this->weights)
		{
			for (uint16_t i = 0; i < this->out_channels; i++)
			{
				for (uint32_t j = 0; j < this->kernel_steps; j++)
				{
					cout << this->get_channel_data(i)[j] << ",";
				}
				cout << endl;
			}
		}
	}
	void Conv_2d::print_bias()
	{
		if (this->bias)
		{
			for (uint16_t i = 0; i < this->out_channels; i++)
				cout << this->bias[i] << ",";
		}
		cout << endl;
	}
	void Conv_2d::to_binay_file(ofstream & outfile)
	{
		outfile.write((char*)&this->ID, sizeof(uint8_t));
		outfile.write((char*)&this->in_channels, sizeof(uint16_t));
		outfile.write((char*)&this->out_channels, sizeof(uint16_t));
		outfile.write((char*)&this->k_w, sizeof(uint8_t));
		outfile.write((char*)&this->k_h, sizeof(uint8_t));
		outfile.write((char*)&this->strid, sizeof(conv_stride));
		outfile.write((char*)&this->padding, sizeof(conv_padding));
		outfile.write((char*)this->weights, this->total_size * DBYTES);
		outfile.write((char*)&this->with_bias, sizeof(bool));
		if (this->with_bias)
			outfile.write((char*)this->bias, this->out_channels*DBYTES);
	}

	void Conv_2d::read_stream(ifstream& instream)
	{
		instream.read((char*)&(this->in_channels), sizeof(uint16_t));
		instream.read((char*)&(this->out_channels), sizeof(uint16_t));
		instream.read((char*)&(this->k_w), sizeof(uint8_t));
		instream.read((char*)&(this->k_h), sizeof(uint8_t));
		instream.read((char*)&(this->strid), sizeof(conv_stride));
		instream.read((char*)&(this->padding), sizeof(conv_padding));

		this->kernel_size = this->k_w*this->k_h;
		this->kernel_steps = this->kernel_size*this->in_channels;
		uint8_t tail = this->kernel_steps % DATA_ALIGN;
		if (tail == 0)
		{
			this->data_pad = 0;
			this->kernel_steps_pad = this->kernel_steps;
		}
		else
		{
			this->data_pad = DATA_ALIGN - tail;
			this->kernel_steps_pad = this->kernel_steps + this->data_pad;
		}
		this->total_size = this->kernel_steps_pad * this->out_channels;
		this->weights = (float*)mylloc(this->total_size * DBYTES, MALLOC_ALIGN);


		instream.read((char*)(this->weights), this->total_size * DBYTES);
		instream.read((char*)&(this->with_bias), sizeof(bool));
		if (this->with_bias)
		{
			this->bias = new DTYPE[this->out_channels]{};
			instream.read((char*)(this->bias), this->out_channels*DBYTES);
		}
	}

	string Conv_2d::info()
	{
		string out = "";
		char data[200];
		sprintf(data, "Operator Conv2d\ninput channels: %d\noutput channels: %d\nkernel size: [%d,%d]\nstrid: [%d,%d]\npadding: [%d,%d,%d,%d]\n",
			this->in_channels,this->out_channels,this->k_w,this->k_h,this->strid.w,this->strid.h,this->padding.left,this->padding.top,
			this->padding.right,this->padding.bottom);
		out += data;
		return out;
	}

	Conv_2d::~Conv_2d()
	{
		this->k_w = 0;
		this->k_h = 0;
		this->in_channels = 0;
		this->out_channels = 0;
		this->total_size = 0;
		this->kernel_size = 0;
		this->kernel_steps = 0;
		this->kernel_steps_pad = 0;
		this->data_pad = 0;
		this->tm_rows = 0;
		this->tm_cols = 0;
		remove_mylloc(weights);
		remove_mylloc(transform_matrix);
		remove_new(bias);
		remove_new(dL_dw);
		remove_new(dL_dw_1);
		remove_new(dL_dw_2);
		remove_new(dL_db);
		remove_new(dL_db_1);
		remove_new(dL_db_2);
		remove_node(dL_din);
		remove_node(dL_dout);
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
		this->ID = LAYER_ID::DENSE;
	}
	Dense::Dense(uint32_t in_size, uint32_t out_size, bool with_bias):
		in_size(in_size),
		out_size(out_size),
		with_bias(with_bias)
	{
		this->ID = LAYER_ID::DENSE;
		uint8_t tail = in_size%DATA_ALIGN;
		if (tail == 0) this->data_pad = 0;
		else this->data_pad = DATA_ALIGN - tail;
		this->in_size_pad = this->in_size + this->data_pad;
		this->total_size = this->in_size_pad*out_size;
		this->weights = (DTYPE*)mylloc(this->total_size*DBYTES, MALLOC_ALIGN);
		if(this->with_bias)
			this->bias = new DTYPE[this->out_size]{};
		else this->bias = nullptr;
	}
	void Dense::random_init(int init_method)
	{
		if (init_method == Uniform)
		{
			for (uint32_t i = 0; i < this->out_size; i++)
			{
				for (uint32_t j = 0; j < this->in_size; j++)
				{
					this->get_channel_data(i)[j] = random_uniform();
				}
				if (this->data_pad > 0)
				{
					for (uint8_t j = 0; j < this->data_pad; j++)
						this->get_channel_data(i)[this->in_size + j] = 0;
				}
			}
		}
		else if (init_method == Normal)
		{
			for (uint32_t i = 0; i < this->out_size; i++)
			{
				for (uint32_t j = 0; j < this->in_size; j++)
				{
					this->get_channel_data(i)[j] = random_gaussrand(0,1);
				}
				if (this->data_pad > 0)
				{
					for (uint8_t j = 0; j < this->data_pad; j++)
						this->get_channel_data(i)[this->in_size + j] = 0;
				}
			}
		}
		else
			throw "Please identify init method(Normal or Uniform)";
	}

	void __dense_async_helper(Node input, Node out,Dense* dense, DTYPE* res, DTYPE* inp, uint32_t start, uint32_t end)
	{
		if(dense->with_bias)
			for (uint32_t j = start; j < end; j++)
				res[j] = MUL_ADD(dense->get_channel_data(j), inp, dense->in_size_pad) + dense->bias[j];
		else
			for (uint32_t j = start; j < end; j++)
				res[j] = MUL_ADD(dense->get_channel_data(j), inp, dense->in_size_pad) + dense->bias[j];
	}

	Node Dense::forward(Node input)
	{
		if (this->in_size_pad != input->batch_steps_pad)
			throw "dims miss match!";
		Node out = Node_creater::creat(input->batchsize, 1, 1, this->out_size);
		if (this->parall_thread > 1 && this->out_size > this->parall_thread)
		{
			future<void>* fn = new future<void>[this->parall_thread];
			uint32_t base_n = this->out_size / this->parall_thread;
			for (uint16_t b = 0; b < input->batchsize; b++)
			{
				DTYPE* res = out->get_batch_data(b);
				DTYPE* inp = input->get_batch_data(b);
				for (uint8_t i = 0; i < this->parall_thread - 1; i++)
				{
					fn[i] = async(launch::async, __dense_async_helper, input, out, this,
						res, inp, base_n*i, base_n*(i + 1));
				}
				fn[this->parall_thread - 1] = async(launch::async, __dense_async_helper, input, out, this,
					res, inp, base_n*(this->parall_thread - 1), this->out_size);
				for (int i = 0; i < this->parall_thread; i++)
					fn[i].wait();
			}
			delete[] fn;
		}
		else
		{
			if (this->with_bias)
			{
				for (uint16_t i = 0; i < input->batchsize; i++)
				{
					DTYPE* res = out->get_batch_data(i);
					DTYPE* inp = input->get_batch_data(i);
					for (uint32_t j = 0; j < this->out_size; j++)
						res[j] = MUL_ADD(this->get_channel_data(j), inp, this->in_size_pad) + this->bias[j];
				}
			}
			else
			{
				for (uint16_t i = 0; i < input->batchsize; i++)
				{
					DTYPE* res = out->get_batch_data(i);
					DTYPE* inp = input->get_batch_data(i);
					for (uint32_t j = 0; j < this->out_size; j++)
						res[j] = MUL_ADD(this->get_channel_data(j), inp, this->in_size_pad);
				}
			}
		}
		this->pre = input->creater;
		out->creater = this;
		return out;
	}

	void Dense::to_binay_file(ofstream& outfile)
	{
		outfile.write((char*)&this->ID, sizeof(uint8_t));
		outfile.write((char*)&this->in_size, sizeof(uint32_t));
		outfile.write((char*)&this->out_size, sizeof(uint32_t));
		outfile.write((char*)&this->with_bias, sizeof(bool));
		outfile.write((char*)this->weights, this->total_size * DBYTES);
		if(this->with_bias)
			outfile.write((char*)this->bias, this->out_size * DBYTES);
	}
	void Dense::read_stream(ifstream& instream)
	{
		instream.read((char*)&(this->in_size), sizeof(uint32_t));
		instream.read((char*)&(this->out_size), sizeof(uint32_t));
		instream.read((char*)&(this->with_bias), sizeof(bool));
		uint8_t tail = this->in_size%DATA_ALIGN;
		if (tail == 0) this->data_pad = 0;
		else this->data_pad = DATA_ALIGN - tail;
		this->in_size_pad = this->in_size + this->data_pad;
		this->total_size = this->in_size_pad*out_size;
		this->weights = (DTYPE*)mylloc(this->total_size*DBYTES, MALLOC_ALIGN);
		instream.read((char*)(this->weights), this->total_size*DBYTES);
		if (this->with_bias)
		{
			this->bias = new DTYPE[this->out_size]{};
			instream.read((char*)(this->bias), this->out_size*DBYTES);
		}
		else this->bias = nullptr;
	}
	string Dense::info()
	{
		string out = "";
		char data[150];
		sprintf(data, "Operator:Dense\ninput channels: %d\noutput channels: %d\n",
			this->in_size, this->out_size);
		out += data;
		return out;

	}
	Dense::~Dense()
	{
		if (this->weights)	myfree(this->weights);
		if (this->bias)	delete[] this->bias;
		this->in_size = 0;
		this->out_size = 0;
		this->with_bias = false;
		this->data_pad = 0;
	}
	Relu::Relu()
	{
		this->ID = LAYER_ID::RELU;
	}

	Node Relu::forward(Node input)
	{
		for (uint16_t i = 0; i < input->batchsize; i++)
		{
			DTYPE* res = input->get_batch_data(i);
			for (uint32_t j = 0; j < input->batch_steps; j++)
				if (res[j] < 0)
					res[j] = 0;
		}
		this->pre - input->creater;
		input->creater = this;
		return input;
	}

	void Relu::to_binay_file(ofstream& outfile)
	{
		outfile.write((char*)&this->ID, sizeof(uint8_t));
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
		this->ID = LAYER_ID::SIGMOID;
	}

	Node Sigmoid::forward(Node input)
	{

		for (uint16_t i = 0; i < input->batchsize; i++)
		{
			DTYPE* res = input->get_batch_data(i);
			for (uint32_t j = 0; j < input->batch_steps; j++)
				res[j] = sigmoid(res[j]);
		}
		this->pre = input->creater;
		input->creater = this;
		return input;
	}

	void Sigmoid::to_binay_file(ofstream& outfile)
	{
		outfile.write((char*)&this->ID, sizeof(uint8_t));
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
		this->ID = LAYER_ID::SOFT_MAX;
	}

	Node Soft_max::forward(Node input)
	{
		for (int i = 0; i < input->batchsize; i++)
		{
			soft_max(input->get_batch_data(i), input->get_batch_data(i), input->batch_steps);
		}
		this->pre = input->creater;
		input->creater = this;
		return input;
	}

	void Soft_max::to_binay_file(ofstream& outfile)
	{
		outfile.write((char*)&this->ID, sizeof(uint8_t));
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
		this->ID = LAYER_ID::BN;
	}
	Batch_normal_2d::Batch_normal_2d(uint16_t channels, DTYPE moment, DTYPE eps, bool with_weights)
		:with_weights(with_weights),
		channels(channels),
		moment(moment),
		eps(eps),
		weight(nullptr),
		bias(nullptr)
	{
		this->ID = LAYER_ID::BN;
		this->cur_mean = new DTYPE[this->channels]{};
		this->cur_var = new DTYPE[this->channels]{};
		this->running_mean = new DTYPE[this->channels]{};
		this->running_var = new DTYPE[this->channels]{};
		if (this->with_weights)
		{
			this->weight = new DTYPE[this->channels]{};
			this->bias = new DTYPE[this->channels]{};
			fill(this->weight, this->weight + this->channels, 1.0f);
		}
	}

	Node Batch_normal_2d::forward(Node input)
	{

		if (this->require_gradients)
		{
			DTYPE temp_M, temp_V;
			for (uint16_t i = 0; i < input->channels; i++)
			{
				temp_M = 0;
				temp_V = 0;
				for (uint16_t j = 0; j < input->batchsize; j++)
				{
					temp_M += SUM(input->get_channel_data(j, i), input->channel_steps) / input->channel_steps / input->batchsize;
				}
				for (uint16_t j = 0; j < input->batchsize; j++)
					temp_V += var_normal(input->get_channel_data(j, i), temp_M, input->channel_steps) / input->batchsize;
				this->running_mean[i] = (1 - this->moment)*this->cur_mean[i] + this->moment*temp_M;
				this->running_var[i] = (1 - this->moment)*this->cur_var[i] + this->moment*temp_V;
				this->cur_mean[i] = temp_M;
				this->cur_var[i] = temp_V;
				for (uint16_t j = 0; j < input->batchsize; j++)
				{
					DTYPE* res = input->get_channel_data(j, i);
					for (uint32_t k = 0; k < input->channel_steps; k++)
					{
						res[k] = (res[k] - temp_M) / sqrt(temp_V + this->eps);
					}
					if (this->with_weights)
						LINEAR_MUL_ADD(res, this->weight[i], this->bias[i], input->channel_steps);
				}
			}
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
						res[k] = (res[k] - this->running_mean[i]) / sqrt(this->running_var[i] + this->eps);
					}
					if (this->with_weights)
						LINEAR_MUL_ADD(res, this->weight[i], this->bias[i], input->channel_steps);
				}
			}
		}
		this->pre = input->creater;
		input->creater = this;
		return input;
	}

	void Batch_normal_2d::to_binay_file(ofstream& outfile)
	{
		outfile.write((char*)&this->ID, sizeof(uint8_t));
		outfile.write((char*)&this->channels, sizeof(uint16_t));
		outfile.write((char*)&this->moment, DBYTES);
		outfile.write((char*)&this->eps, DBYTES);
		outfile.write((char*)&this->with_weights, sizeof(bool));
		outfile.write((char*)this->running_mean, this->channels*DBYTES);
		outfile.write((char*)this->running_var, this->channels*DBYTES);
		if (this->with_weights)
		{
			outfile.write((char*)this->weight, this->channels*DBYTES);
			outfile.write((char*)this->bias, this->channels*DBYTES);
		}

	}

	void Batch_normal_2d::read_stream(ifstream& instream)
	{
		instream.read((char*)&(this->channels), sizeof(uint16_t));
		instream.read((char*)&(this->moment), DBYTES);
		instream.read((char*)&(this->eps), DBYTES);
		instream.read((char*)&(this->with_weights), sizeof(bool));
		this->running_mean = new DTYPE[this->channels]{};
		this->running_var = new DTYPE[this->channels]{};
		instream.read((char*)this->running_mean, this->channels*DBYTES);
		instream.read((char*)this->running_var, this->channels*DBYTES);
		if (this->with_weights)
		{
			this->weight = new DTYPE[this->channels]{};
			this->bias = new DTYPE[this->channels]{};
			instream.read((char*)this->weight, this->channels*DBYTES);
			instream.read((char*)this->bias, this->channels*DBYTES);
		}
	}

	string Batch_normal_2d::info()
	{
		string out = "";
		char data[150];
		sprintf(data, "Operator:Batch_normal_2d\nnchannels: %d\nmoment: %f\neps: %f\n",
			this->channels, this->moment, this->eps);
		out += data;
		return out;
	}

	Batch_normal_2d::~Batch_normal_2d()
	{
		if (this->weight) delete[] this->weight;
		if (this->bias) delete[] this->bias;
		if (this->running_mean) delete[] this->running_mean;
		if (this->running_var) delete[] this->running_var;
		if (this->cur_mean) delete[] this->cur_mean;
		if (this->cur_var) delete[] this->cur_var;
	}

	Max_pool::Max_pool(uint8_t w, uint8_t h) :
		pool_w(w),
		pool_h(h) 
	{
		this->ID = LAYER_ID::MAX_POOL;
	}

	Node Max_pool::forward(Node input)
	{
		Node out = Node_creater::creat(input->batchsize, input->channels, input->rows / this->pool_h, input->cols / this->pool_w);
		for (uint16_t b = 0; b < input->batchsize; b++)
		{
			for (uint16_t c = 0; c < input->channels; c++)
			{
				DTYPE* res = out->get_channel_data(b, c);
				DTYPE* inp = input->get_channel_data(b, c);
				for (uint16_t i = 0, ii = 0; i < input->rows; i += this->pool_h, ii++)
				{
					uint16_t out_row_loc = ii*out->cols;
					uint16_t inp_row_loc = i*input->cols;
					for (uint16_t j = 0, jj = 0; j < input->cols; j += this->pool_w, jj++)
					{
						DTYPE max = inp[inp_row_loc + j];
						for (uint8_t k = 0; k < this->pool_h; k++)
						{
							uint16_t inp_row_loc_ = (i + k)*input->cols;
							for (uint8_t m = 0; m < this->pool_w; m++)
							{
								if (inp[inp_row_loc_ + j + m ] > max)	max = inp[inp_row_loc_ + j + m];
							}
						}
						res[out_row_loc + jj] = max;
					}
				}
			}
		}
		this->pre = input->creater;
		out->creater = this;
		return out;
	}

	void Max_pool::to_binay_file(ofstream& outfile)
	{
		outfile.write((char*)&this->ID, sizeof(uint8_t));
		outfile.write((char*)&(this->pool_w), sizeof(uint8_t));
		outfile.write((char*)&(this->pool_h), sizeof(uint8_t));
	}

	void Max_pool::read_stream(ifstream& instream)
	{
		instream.read((char*)&(this->pool_w), sizeof(uint8_t));
		instream.read((char*)&(this->pool_h), sizeof(uint8_t));
	}

	string Max_pool::info()
	{
		string out = "";
		char data[150];
		sprintf(data, "Operator:Max_pool\nPool_size: [%d,%d]\n",
			this->pool_w, this->pool_h);
		out += data;
		return out;
	}

	Ave_pool::Ave_pool(uint8_t w, uint8_t h): pool_w(w),pool_h(h)
	{
		this->ID = LAYER_ID::AVE_POOL;
	}

	Node Ave_pool::forward(Node input)
	{
		Node out = Node_creater::creat(input->batchsize, input->channels, input->rows / this->pool_h, input->cols / this->pool_w);
		for (uint16_t b = 0; b < input->batchsize; b++)
		{
			for (uint16_t c = 0; c < input->channels; c++)
			{
				DTYPE* res = out->get_channel_data(b, c);
				DTYPE* inp = input->get_channel_data(b, c);
				for (uint16_t i = 0, ii = 0; i < input->rows; i += this->pool_h, ii++)
				{
					uint16_t out_row_loc = ii*out->cols;
					for (uint16_t j = 0, jj = 0; j < input->cols; j += this->pool_w, jj++)
					{
						DTYPE sum = 0;
						for (uint8_t k = 0; k < this->pool_h; k++)
						{
							uint16_t inp_row_loc = (i + k)*input->cols;
							for (uint8_t m = 0; m < this->pool_w; m++)
							{
								sum += inp[inp_row_loc + j + m];
							}
						}
						res[out_row_loc+jj] = sum/this->pool_h/this->pool_w;
					}
				}
			}
		}
		this->pre = input->creater;
		out->creater = this;
		return out;
	}

	void Ave_pool::to_binay_file(ofstream& outfile)
	{
		outfile.write((char*)&this->ID, sizeof(uint8_t));
		outfile.write((char*)&(this->pool_w), sizeof(uint8_t));
		outfile.write((char*)&(this->pool_h), sizeof(uint8_t));
	}


	void Ave_pool::read_stream(ifstream& instream)
	{
		instream.read((char*)&(this->pool_w), sizeof(uint8_t));
		instream.read((char*)&(this->pool_h), sizeof(uint8_t));
	}

	string Ave_pool::info()
	{
		string out = "";
		char data[150];
		sprintf(data, "Operator:Ave_pool\nPool_size: [%d,%d]\n",
			this->pool_w, this->pool_h);
		out += data;
		return out;
	}

	Drop_out::Drop_out(float rate):rate(rate) 
	{
		this->ID = LAYER_ID::DROP_OUT;
	}

	Node Drop_out::forward(Node input)
	{
		srand(clock());
		for (uint16_t i = 0; i < input->batchsize; i++)
			for (uint32_t j = 0; j < input->batch_steps; j++)
			{
				if (random_uniform() <= rate)
					input->get_batch_data(i)[j] = 0;
			}
		this->pre = input->creater;
		input->creater = this;
		return input;
	}

	void Drop_out::to_binay_file(ofstream& outfile)
	{
		outfile.write((char*)&this->ID, sizeof(uint8_t));
		outfile.write((char*)&this->rate, DBYTES);
	}

	void Drop_out::read_stream(ifstream& instream)
	{
		instream.read((char*)&this->rate, DBYTES);
	}

	string Drop_out::info()
	{
		string out = "";
		char data[150];
		sprintf(data, "Operator:Drop_out\nDrop_rate: %f\n",
			this->rate);
		out += data;
		return out;
	}
	Operator*& Concator::get_O1()
	{
		return this->O1;
	}
	Operator*& Concator::get_O2()
	{
		return this->O2;
	}

	void Concator::backward(Optimizer& opt)
	{
		this->set_gradients();
		unordered_set<Operator*> set;
		vector<Operator*> stack;
		stack.push_back(this->O1);
		stack.push_back(this->O1);
		Operator* now;
		while(true)
		{
			now = stack[stack.size() - 1];
			stack.pop_back();
			while (now->get_count_out() <= 1)
			{
				if (now->ID == LAYER_ID::CONCAT)
				{
					stack.push_back(((Concator*)now)->O1);
					stack.push_back(((Concator*)now)->O1);
					break;
				}
				else
				{
					now->backward(opt);
					now = now->pre;
				}
			}
			if (now->ID == LAYER_ID::CONCAT)
				continue;
			if (set.count(now) == 0)
			{
				now->increase_count_back();
				set.insert(now);
			}
			else
			{
				now->increase_count_back();
				if (now->get_count_back() == now->get_count_out())
				{
					if (stack.size() > 0)
					{
						stack.push_back(now);
						set.erase(now);
					}
					else
					{
						this->start = now;
						break;
					}
				}
			}
		}
	}

	Operator* Concator::get_pre()
	{
		return this->start;
	}

	Add::Add(Operator* O1, Operator* O2)
	{
		this->ID = LAYER_ID::CONCAT;
		this->O1 = O1;
		this->O2 = O2;
	}

	void Add::set_gradients()
	{
		this->O1->pass_gradients(this->dL_dout);
		this->O2->pass_gradients(this->dL_dout);
	}
}