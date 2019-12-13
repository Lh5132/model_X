#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace model_X
{
	using namespace std;
	class Conv_2d;
	class Dense;
	class Batch_normal_2d;
	class Drop_out;
	class Max_pool;
	class Ave_pool;
	class node;
	typedef shared_ptr<node> Node;
	void node_to_cuda(node* input);
	void node_to_cpu(node* input);

	void conv_forward(Node& input, Node& out, Conv_2d*& conv);
	void conv_backend(Node& input, Node& out, Conv_2d*& conv);
	void conv_to_cuda(Conv_2d*& conv);
	void conv_to_cpu(Conv_2d*& conv);
}