#pragma once
#include "storage.h"
#include "layer.h"
#include "optimizer.h"

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
		void backward(Optimizer::base_optimizer& opt);
	};

	namespace Loss_Functions
	{
		Loss BCELoss(const tensor& output, const tensor& ground_truth);
		Loss MSELoss(const tensor& output, const tensor& ground_truth);
		Loss SOFTMAXLoss(const tensor& output, const tensor& ground_truth);
		Loss BCEWithLogitsLoss(const tensor& output, const tensor& ground_truth);
		Loss MSEWithLogitsLoss(const tensor& output, const tensor& ground_truth);
		Loss SOFTMAXWithLogitsLoss(const tensor& output, const tensor& ground_truth);
	}
}