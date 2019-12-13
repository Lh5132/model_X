#pragma once

#include "util.h"
#include <fstream>
#include <memory>


namespace model_X
{
	class Operator;
	class node
	{
	public:
		uint16_t rows;
		uint16_t cols;
		uint16_t channels;
		uint16_t batchsize;
		uint32_t channel_steps;
		uint32_t batch_steps;
		uint32_t batch_steps_pad;
		uint32_t total_size;
		uint8_t data_pad;
		DTYPE* data;
		node* dout_dthis = nullptr;
		Operator* creater;    //记录每一层的信息
		bool require_gradients = false;

		node();
		node(uint16_t batchsize, uint16_t channels, uint16_t rows, uint16_t cols, bool require_gradients = false);
		
#ifdef CUDA
		node* cuda_data; //用于管理GPU中的node数据
		bool is_cuda;
		void to_cuda();
		void to_cpu();
#endif
		void random_init(int init_method = Uniform);
		node* copy(bool with_data = true);
		void map(DTYPE(*pf)(DTYPE));
		void print_shape();
		void print_data();
		~node();

		//内联查询函数
		inline DTYPE* get_batch_data(uint16_t batch)
		{
			return this->data + batch*this->batch_steps_pad;
		}
		inline DTYPE* get_channel_data(uint16_t batch, uint16_t channel)
		{
			return this->data + batch*this->batch_steps_pad + channel*this->channel_steps;
		}
		inline DTYPE& at(uint16_t batch, uint16_t channel, uint16_t row, uint16_t col)
		{
			return this->data[batch*this->batch_steps_pad + channel*this->channel_steps + row*this->cols + col];
		}
		inline void set_zero()
		{
			fill(this->data, this->data + this->total_size, 0.0f);
		}
		inline void set_one()
		{
			fill(this->data, this->data + this->total_size, 1.0f);
		}
		inline void require_grad()
		{
			this->require_gradients = true;
		}
		inline void free_grad()
		{
			this->require_gradients = false;
		}
	};
	class Node final : public shared_ptr<node>
	{
//	public:
//		uint16_t get_rows() const;
//		uint16_t get_cols() const;
//		uint16_t get_channels() const;
//		uint16_t get_batch_size() const;
//		uint16_t get_data_size() const;
//		uint32_t get_channel_steps() const;
//		uint32_t get_batch_steps() const;
//		uint32_t get_batch_steps_pad() const;
//		uint8_t get_data_pad() const;
//#ifdef CUDA
//		node* get_cuda_data; //用于管理GPU中的node数据
//		void to_cuda();
//		void to_cpu();
//#endif
	public:
		Node operator+ (Node other);
		Node operator- (Node other);
		Node operator* (Node other);
		Node operator/ (Node other);
		Node operator^ (Node other);
		Node operator+ (DTYPE other);
		Node operator- (DTYPE other);
		Node operator* (DTYPE other);
		Node operator/ (DTYPE other);
		Node operator^ (DTYPE other);
	};
	class Node_creater
	{
	public:
		static Node creat(uint16_t batchsize, uint16_t channels, uint16_t rows, uint16_t cols, bool reqquire_gradients = false);
	};
}
