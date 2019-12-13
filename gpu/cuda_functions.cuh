#pragma once

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
	class Node;
	void node_to_cuda(node* input);
	void node_to_cpu(node* input);

	void cuda_conv_forward(Node& input, Node& out, Conv_2d* conv);
	void cuda_conv_backend(Node& input, Node& out, Conv_2d* conv);
	void conv_to_cuda(Conv_2d* conv);
	void conv_to_cpu(Conv_2d* conv);
}