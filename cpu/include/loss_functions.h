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
		Loss BCELoss(const storage& output, const storage& ground_truth);
		Loss MSELoss(const storage& output, const storage& ground_truth);
		Loss SOFTMAXLoss(const storage& output, const storage& ground_truth);
		Loss BCEWithLogitsLoss(const storage& output, const storage& ground_truth);
		Loss MSEWithLogitsLoss(const storage& output, const storage& ground_truth);
		Loss SOFTMAXWithLogitsLoss(const storage& output, const storage& ground_truth);
	}
}