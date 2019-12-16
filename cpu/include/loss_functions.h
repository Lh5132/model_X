#pragma once
#include "node.h"
#include "layer.h"

namespace model_X
{
	class Loss
	{
	private:
		Operator* backward_start;
		DTYPE loss;
	public:
		Loss(Operator* backward_start, DTYPE loss);
		Loss(const Loss& other);
		DTYPE item();
		void backward(Optimizer& opt);
	};

	namespace Loss_Functions
	{
		Loss BCELoss(const Node& output, const Node& ground_truth);
		Loss MSELoss(const Node& output, const Node& ground_truth);
		Loss SOFTMAXLoss(const Node& output, const Node& ground_truth);
		Loss BCEWithLogitsLoss(const Node& output, const Node& ground_truth);
		Loss MSEWithLogitsLoss(const Node& output, const Node& ground_truth);
		Loss SOFTMAXWithLogitsLoss(const Node& output, const Node& ground_truth);
	}
}