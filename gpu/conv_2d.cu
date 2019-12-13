#include "cuda_functions.cuh"
#include "layer.h"
#include "node.h"

namespace model_X
{
	__global__ void conv_forward_helper(node* input, node* out, Conv_2d* conv)
	{
		uint16_t out_row = threadIdx.x;
		uint16_t out_col = threadIdx.y;
		uint16_t batch = blockIdx.x;
		uint16_t channel = blockIdx.y;
		int ii = out_row*conv->strid.h - conv->padding.top;
		int jj = out_col*conv->strid.w - conv->padding.left;
		DTYPE* kernel = conv->weights + channel*conv->kernel_steps_pad;
		DTYPE* inp = input->data + batch*input->batch_steps_pad;
		DTYPE sum = 0;
		for (uint8_t krow = 0; krow < conv->k_h; krow++)
		{
			for (uint8_t kcol = 0; kcol < conv->k_w; kcol++)
			{
				int input_row = ii + krow;
				int input_col = jj + kcol;
				if (input_row >= 0 && input_row < input->rows&&input_col >= 0 && input_col < input->cols)
				{
					for (uint16_t c = 0; c < input->channels; c++)
					{
						sum += inp[c*input->channel_steps + input_row*input->cols + input_col] * \
							kernel[c*conv->kernel_size+krow*conv->k_w+kcol];
					}
				}
			}
		}
		out->data[batch*out->batch_steps_pad + channel*out->channel_steps + out_row * out->cols + out_col] = sum;
	}
	__global__ void conv_backend_helper(node* input, node* out, Conv_2d* conv)
	{

	}
	void cuda_conv_forward(Node& input, Node& out, Conv_2d* conv)
	{
		dim3 grid(out->batchsize, out->channels);
		dim3 block(out->rows, out->cols);
		conv_forward_helper << <grid, block >> >(input->cuda_data, out->cuda_data, conv->cuda_data);
	}

	void cuda_conv_backend(Node& input, Node& out, Conv_2d* conv)
	{

	}

	void conv_to_cuda(Conv_2d* conv)
	{
		DTYPE* cuda_weights = 0;
		DTYPE* cuda_bias = 0;
		//将conv中的所有数据拷贝至GPU中，同时指定cuda_data指向gpu中的该数据
		cudaMalloc((void**)&conv->cuda_data, sizeof(Conv_2d));
		cudaMalloc((void**)&cuda_weights, conv->total_size * DBYTES);
		cudaMemcpy(conv->cuda_data, conv, sizeof(Conv_2d), cudaMemcpyHostToDevice);
		cudaMemcpy(cuda_weights, conv->weights, conv->total_size * DBYTES, cudaMemcpyHostToDevice);
		cudaMemcpy(&conv->cuda_data->weights, &cuda_weights, sizeof(void*), cudaMemcpyHostToDevice);
		if (conv->with_bias)
		{
			cudaMalloc((void**)&cuda_bias, conv->out_channels * DBYTES);
			cudaMemcpy(cuda_bias, conv->bias, conv->total_size * DBYTES, cudaMemcpyHostToDevice);
			cudaMemcpy(&conv->cuda_data->bias, &cuda_bias, sizeof(void*), cudaMemcpyHostToDevice);
		}
	}
	/*
	将GPU中的数据拷贝出来
	*/
	void conv_to_cpu(Conv_2d* conv)
	{
		DTYPE* cuda_weights = 0;
		DTYPE* cuda_bias = 0;
		cudaMemcpy(&cuda_weights, &conv->cuda_data->weights, sizeof(void*), cudaMemcpyHostToDevice);
		cudaMemcpy(conv->weights, cuda_weights, conv->total_size*DBYTES, cudaMemcpyHostToDevice);
		if (conv->with_bias)
		{
			cudaMemcpy(&cuda_bias, &conv->cuda_data->bias, sizeof(void*), cudaMemcpyHostToDevice);
			cudaMemcpy(conv->bias, cuda_bias, conv->out_channels*DBYTES, cudaMemcpyHostToDevice);
		}
		cudaFree(conv->cuda_data);
	}
}