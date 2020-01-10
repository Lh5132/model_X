#pragma once

#include "util.h"
#include <fstream>

#ifdef CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#endif



namespace model_X
{
	const int cpu_cors = get_cpu_cors();
	namespace Optimizer
	{
		class base_optimizer;
		class SGD;
		class Momentum;
		class RMSProp;
		class Adam;
	}
	namespace PADDING_STYLE
	{
		static const uint8_t SAME = 1;
		static const uint8_t NONE = 2;
	}
	namespace LAYER_ID
	{
		static const uint8_t CONV_2D = 1;
		static const uint8_t DENSE = 2;
		static const uint8_t CONCAT = 3;
		static const uint8_t RELU = 4;
		static const uint8_t SIGMOID = 5;
		static const uint8_t SOFT_MAX = 6;
		static const uint8_t BN = 7;
		static const uint8_t MAX_POOL = 8;
		static const uint8_t AVE_POOL = 9;
		static const uint8_t DROP_OUT = 10;
	}
	typedef struct conv_stride
	{
		uint8_t w;
		uint8_t h;
		conv_stride() {
			this->w = 1;
			this->h = 1;
		};
		conv_stride(uint8_t w, uint8_t h) {
			this->w = w;
			this->h = h;
		}
	} conv_stride;
	typedef struct conv_padding
	{
		uint8_t left;
		uint8_t top;
		uint8_t right;
		uint8_t bottom;
		uint8_t Padding_style;
		conv_padding() {
			this->Padding_style = 0;
			this->left = 0;
			this->top = 0;
			this->right = 0;
			this->bottom = 0;
		};
		conv_padding(uint8_t Padding_style)
		{
			this->Padding_style = Padding_style;
			this->left = 0;
			this->top = 0;
			this->right = 0;
			this->bottom = 0;
		}
		conv_padding(uint8_t l, uint8_t t, uint8_t r, uint8_t b) {
			this->Padding_style = 0;
			this->left = l;
			this->top = t;
			this->right = r;
			this->bottom = b;
		}
	} conv_padding;

	class storage;
	class tensor;
	class Operator
	{
	protected:
		bool require_gradients = false;
		uint8_t count_out = 0; //用于记录分支开始的节点
		uint8_t count_back = 0;

		//导数传递
		int parall_thread = 0;

		DTYPE* weights;
		DTYPE* bias;
		uint32_t weights_size;
		uint32_t bias_size;
		DTYPE* dL_dw_now = nullptr;
		DTYPE* dL_db_now = nullptr;
		DTYPE* dL_dw = nullptr;
		DTYPE* dL_db = nullptr;
		DTYPE* dL_dw_1 = nullptr;
		DTYPE* dL_db_1 = nullptr;
		DTYPE* dL_dw_2 = nullptr;
		DTYPE* dL_db_2 = nullptr;

	public:
		friend class Optimizer::SGD;
		friend class Optimizer::Momentum;
		friend class Optimizer::RMSProp;
		friend class Optimizer::Adam;

		uint8_t ID = 0;
		Operator* pre = nullptr;   //用于记录从后向前的单向链表
		/*
		dc_dout:	损失对该层输出的导数矩阵
		dout_din:	该层输出对输入的导数矩阵，用于链式求解前一层的dc_dout;
		*/
		storage* dL_dout = nullptr;
		storage* dL_din = nullptr;
		inline bool is_gradiets()
		{
			return this->require_gradients;
		}

		inline void increase_count_out()
		{
			this->count_out += 1;
		}
		inline void increase_count_back()
		{
			this->count_back += 1;
		}
		inline int get_count_out()
		{
			return this->count_out;
		}
		inline int get_count_back()
		{
			return this->count_back;
		}
		inline void eval()
		{
			this->require_gradients = false;
		}
		inline void train()
		{
			this->require_gradients = true;
		}
		void pass_gradients();
		void pass_gradients(storage* gradients);
		Operator* get_pre();
		/*
		计算图通过一个单向的带有分支的链表来表示
		当某一Operator的count_out>1时，该节点即为某一分支开始分裂的起点
		所有分支通过Concator来进行合并（有+、-、*、/ 共四种Concator），（例如Resnet中的残差链接）
		*/
		virtual void set_async_thread(int n);
		virtual void set_async_thread();
		virtual tensor forward(tensor& input) = 0;
		virtual void backward(Optimizer::base_optimizer& opt);
		virtual void zero_grad();
		virtual void to_binay_file(ofstream& outfile);
		virtual string info();
		virtual void random_init(int init_method = Normal);
		virtual ~Operator();

#ifdef CUDA
		bool is_cuda = false;
		virtual void to_cuda();
		virtual void to_cpu();
#endif
	};

	//为了方便使用avx2，所有数据均用0补成8的整数倍的数量
	class Conv_2d final:public Operator
	{
	private:
		//卷积核参数
		uint8_t k_w;
		uint8_t k_h;
		uint16_t in_channels;
		uint16_t out_channels;
		uint16_t kernel_size;
		uint32_t total_size;
		uint32_t kernel_steps;
		bool with_bias;

		conv_stride strid;
		conv_padding padding;
		//转换矩阵参数
		uint32_t tm_rows;
		uint32_t tm_cols;
		uint32_t tm_batch_steps;
		//输出对卷积参数的雅可比矩阵
		DTYPE* dout_dw = nullptr;

		//输出对输入的雅可比矩阵，该矩阵为稀疏矩阵
		uint16_t* dout_din_w_loc = nullptr;				
		uint16_t* dout_din_out_loc = nullptr;
		uint16_t* dout_din_row_size = nullptr;

		//记录误差项对卷积参数的梯度值(一阶和二阶)的动量平均，在Adam中将会用到
		uint32_t time_step = 0;

#ifdef CUDA
	private:
		Conv_2d* cuda_data; //用于管理在gpu内存中的卷积
	public:
		friend void conv_to_cuda(Conv_2d* conv);
		friend void conv_to_cpu(Conv_2d* conv);
		friend __global__ void conv_forward_helper(storage* input, storage* out, Conv_2d* conv);
		friend void cuda_conv_forward(storage& input, storage& out, Conv_2d* conv);
		void to_cuda();
		void to_cpu();
#endif
	public:
		friend void __conv_async_helper(storage* input, storage* out, Conv_2d* conv,
			uint32_t start, uint32_t end);

		inline DTYPE* get_channel_data(uint16_t channel)
		{
			return this->weights + channel * this->kernel_steps;
		}
		inline DTYPE* get_bias()
		{
			return this->bias;
		}
		Conv_2d();
		Conv_2d(uint16_t in_channels, uint16_t out_channels, uint8_t w, uint8_t h, 
			conv_stride strid, conv_padding padding, bool with_bias = true);
		tensor forward(tensor& input) override;
		void random_init(int init_method = Normal) override;
		void zero_grad() override;
		void backward(Optimizer::base_optimizer& opt) override;
		void print_weight();
		void print_bias();
		//模型IO函数
		void to_binay_file(ofstream& outfile) override;
		void read_stream(ifstream& instream);
		string info() override;
		~Conv_2d() override;
	};

	class Dense final :public Operator
	{
	private:
		uint32_t in_size;
		uint32_t out_size;
		uint32_t total_size;
		bool with_bias;
		storage* dout_dw = nullptr;
		uint32_t time_step = 0;
	public:
		friend void __dense_async_helper(Dense* dense, DTYPE* res, DTYPE* inp, uint32_t start, uint32_t end);
		Dense();
		Dense(uint32_t in_size, uint32_t out_size, bool with_bias = true);
		void random_init(int init_method = Normal) override;
		inline DTYPE* get_channel_data(uint16_t c)
		{
			return this->weights + c * this->in_size;
		}
		tensor forward(tensor& input) override;
		void backward(Optimizer::base_optimizer& opt) override;
		void to_binay_file(ofstream& outfile) override;
		void read_stream(ifstream& instream);
		string info() override;
		~Dense() override;
	};
	class Relu final: public Operator
	{
	public:
		Relu();
		tensor forward(tensor& input) override;
		void backward(Optimizer::base_optimizer& opt) override;
		void to_binay_file(ofstream& outfile) override;
		void read_stream(ifstream& instream);
		string info() override;
	};
	class Sigmoid final: public Operator
	{
	public:
		Sigmoid();
		tensor forward(tensor& input) override;
		void backward(Optimizer::base_optimizer& opt) override;
		void to_binay_file(ofstream& outfile) override;
		void read_stream(ifstream& instream);
		string info() override;
	};
	class Soft_max final: public Operator
	{
	public:
		Soft_max();
		tensor forward(tensor& input) override;
		void backward(Optimizer::base_optimizer& opt) override;
		void to_binay_file(ofstream& outfile) override;
		void read_stream(ifstream& instream);
		string info() override;
	};
	class Batch_normal_2d final : public Operator
	{
	private:
		uint16_t channels;
		bool with_weights;
		DTYPE moment;
		DTYPE eps;
		DTYPE* cur_mean;
		DTYPE* cur_var;
		DTYPE* running_mean;
		DTYPE* running_var;

		storage* dout_dw = nullptr;
		DTYPE* dout_din = nullptr;

	public:
		Batch_normal_2d();
		Batch_normal_2d(uint16_t channels, DTYPE moment = 0.1, DTYPE eps = 1e-5, bool with_weights = true);
		tensor forward(tensor& input) override;
		void backward(Optimizer::base_optimizer& opt) override;
		void to_binay_file(ofstream& outfile) override;
		void read_stream(ifstream& instream);

		string info() override;

		~Batch_normal_2d() override;
	};
	class Max_pool final :public Operator
	{
	public:
		uint8_t pool_w;
		uint8_t pool_h;
		Max_pool(uint8_t w = 2, uint8_t h = 2);
		tensor forward(tensor& input) override;
		void backward(Optimizer::base_optimizer& opt) override;
		void to_binay_file(ofstream& outfile) override;
		void read_stream(ifstream& instream);

		string info() override;
	};
	class Ave_pool final :public Operator
	{
	public:
		uint8_t pool_w;
		uint8_t pool_h;
		Ave_pool(uint8_t w = 2, uint8_t h = 2);
		tensor forward(tensor& input) override;
		void backward(Optimizer::base_optimizer& opt) override;
		void to_binay_file(ofstream& outfile) override;
		void read_stream(ifstream& instream);
		string info() override;
	};
	class Drop_out final : public Operator
	{
	public:
		DTYPE rate;
		Drop_out(float rate = 0.5);
		tensor forward(tensor& input) override;
		void backward(Optimizer::base_optimizer& opt) override;
		void to_binay_file(ofstream& outfile) override;
		void read_stream(ifstream& instream);
		string info() override;
	};

	class Concator :public Operator
	{
	protected:
		Operator* O1;
		Operator* O2;
		Operator* start = nullptr;
	public:
		Operator*& get_O1();
		Operator*& get_O2();
		tensor forward(tensor& input) override;
		virtual void set_gradients() = 0;
	};

	class Add final : public Concator
	{
	public:
		Add(Operator* O1, Operator* O2);
		void set_gradients() override;
	};
}

