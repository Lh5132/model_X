#define CU
#include "cuda_functions.cuh"
#include "node.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace model_X
{
	void node_to_cuda(node* input)
	{
		DTYPE* temp_data = 0;
		size_t t_size = input->total_size*DBYTES;
		cudaMalloc((void**)&input->cuda_data, sizeof(node));
		cudaMalloc((void**)&temp_data, t_size);
		cudaMemcpy(input->cuda_data, input, sizeof(node), cudaMemcpyHostToDevice);
		cudaMemcpy(temp_data, input->data, t_size, cudaMemcpyHostToDevice);
		cudaMemcpy(&input->cuda_data->data, &temp_data, sizeof(void*), cudaMemcpyHostToDevice);
	}
	void node_to_cpu(node* input)
	{
		DTYPE* temp_data = 0;
		cudaMemcpy(&temp_data, &input->cuda_data->data, sizeof(void*), cudaMemcpyDeviceToHost);
		cudaMemcpy(input->data, temp_data, input->total_size*DBYTES, cudaMemcpyDeviceToHost);
		cudaFree(input->cuda_data);
	}
}