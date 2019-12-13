#ifdef CUDA
#include "../../gpu/cuda_functions.cuh"
#endif

#include "node.h"
#include <iostream>
#include <string.h>
#include "layer.h"

namespace model_X
{
	node::node() :
		batchsize(0),
		channels(0),
		cols(0),
		rows(0),
		channel_steps(0),
		batch_steps(0),
		batch_steps_pad(0),
		total_size(0),
		creater(nullptr),
		data(nullptr) {}
	node::node(uint16_t batchsize, uint16_t channels, uint16_t rows, uint16_t cols, bool require_gradients)
		:batchsize(batchsize),
		rows(rows),
		cols(cols),
		channels(channels),
		creater(nullptr),
		require_gradients(require_gradients),
		channel_steps(rows*cols)
	{
		this->batch_steps = this->channel_steps*channels;
		uint8_t tail = this->batch_steps % DATA_ALIGN;
		if (tail == 0)	this->data_pad = 0;
		else
		{
			this->data_pad = DATA_ALIGN - tail;
		}
		this->batch_steps_pad = this->batch_steps + this->data_pad;
		this->total_size = this->batch_steps_pad * batchsize;
		
		this->data = (float*)mylloc(this->total_size * DBYTES, MALLOC_ALIGN);
		if (tail > 0)
		{
			for (int i = 0; i < this->batchsize; i++)
			{
				for (int j = 0; j < this->data_pad; j++)
					this->data[i*this->batch_steps_pad + this->batch_steps + j] = 0;
			}
		}
	}

#ifdef CUDA
	void node::to_cuda()
	{
		if (!this->is_cuda)
		{
			this->is_cuda = true;
			node_to_cuda(this);
		}
	}
	void node::to_cpu()
	{
		if (this->is_cuda)
		{
			this->is_cuda = false;
			node_to_cpu(this);
		}
	}
#endif

	void  node::random_init(int init_method)
	{
		if (init_method == Uniform)
		{
			for (uint16_t i = 0; i < this->batchsize; i++)
			{
				for (uint32_t j = 0; j < this->batch_steps; j++)
					this->data[i*this->batch_steps_pad + j] = random_uniform();
			}
		}
		else if (init_method == Normal)
		{
			for (uint16_t i = 0; i < this->batchsize; i++)
			{
				for (uint32_t j = 0; j < this->batch_steps; j++)
					this->data[i*this->batch_steps_pad + j] = random_gaussrand(0,1);
			}
		}
		else
			throw "Please identify init method(Normal or Uniform)";
	}

	node * node::copy(bool with_data)
	{
		node* out = new node(this->batchsize, this->channels, this->rows, this->cols,this->require_gradients);
		if(with_data)
			memcpy(out->data, this->data, this->total_size * DBYTES);
		return out;
	}

	void node::map(DTYPE(*pf)(DTYPE))
	{
		for (uint16_t i = 0; i < this->batchsize; i++)
		{
			for (uint32_t j = 0; j < this->batch_steps; j++)
				this->data[i*this->batch_steps_pad + j] = pf(this->data[i*this->batch_steps + j]);
		}
	}

	void node::print_shape()
	{
		cout << "[" << this->batchsize << ", " << this->channels << ", " << this->rows << ", " << this->cols << "]" << endl;
	}

	void node::print_data()
	{
		for (uint16_t b = 0; b < this->batchsize; b++)
		{
			for (uint16_t c = 0; c < this->channels; c++)
			{
				for (uint16_t i = 0; i < this->rows; i++)
				{
					for (uint16_t j = 0; j < this->cols; j++)
					{
						cout << this->at(b, c, i, j) << ",";
					}
					cout << endl;
				}
			}
		}
	}
	node::~node()
	{
		this->batchsize = 0;
		this->channels = 0;
		this->cols = 0;
		this->rows = 0;
		this->channel_steps = 0;
		this->batch_steps = 0;
		this->batch_steps_pad = 0;
		this->total_size = 0;
		this->creater = nullptr;
		if(this->data)
			myfree(this->data);
		if (this->dout_dthis)
			this->dout_dthis->~node();
	}

	Node Node::operator+ (Node other)
	{
		Node out = Node_creater::creat(other->batchsize, other->channels, other->rows, other->cols);
		ADD((*this)->data, other->data, out->data,other->total_size);
		if ((*this)->require_gradients)
		{
			if ((*this)->creater && other->creater)
			{
				Add *s = new Add((*this)->creater, other->creater);
				out->creater = s;
				(*this)->creater->increase_count_out();
				other->creater->increase_count_out();
			}
			//if ((*this)->creater && !other->creater)
			//{
			//	out->creater = (*this)->creater;
			//	out->dout_dthis = (*this)->copy(false);
			//	out->dout_dthis->set_one();
			//}
			//if (!(*this)->creater && other->creater)
			//{
			//	out->creater = other->creater;
			//	out->dout_dthis = (*this)->copy(false);
			//	out->dout_dthis->set_one();
			//}
		}
		return out;
	}



	Node Node_creater::creat(uint16_t batchsize, uint16_t channels, uint16_t rows, uint16_t cols, bool reqquire_gradients)
	{
		_Ref_count_obj<node> *_Rx =
			new _Ref_count_obj<node>(batchsize, channels, rows, cols, reqquire_gradients);

		Node _Ret;
		_Ret._Resetp0(_Rx->_Getptr(), _Rx);
		return _Ret;
	}

}