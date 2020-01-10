#include "optimizer.h"
namespace model_X
{
	namespace Optimizer
	{
		SGD::SGD(DTYPE lr)
		{
			this->lr = lr;
		}
		void SGD::apply_gradients(Operator* op)
		{
			if(op->bias)
				APPLY_GRADIENTS(op->dL_db_now, op->bias, lr, op->bias_size);
			APPLY_GRADIENTS(op->dL_dw_now, op->weights, lr, op->weights_size);
		}

		Momentum::Momentum(DTYPE lr, DTYPE momentum) :momentum(momentum) 
		{
			this->lr = lr;
		}
		void Momentum::apply_gradients(Operator* op)
		{
			if (!op->dL_dw)
			{
				op->dL_dw = new DTYPE[op->weights_size]{};
			}
			if (op->bias)
			{
				if (!op->dL_db)
					op->dL_db = new DTYPE[op->bias_size]{};
				for (uint32_t i = 0; i < op->weights_size; i++)
				{
					op->dL_dw[i] = op->dL_dw[i] * momentum + (1 - momentum) * op->dL_dw_now[i];
					if (i < op->bias_size)
						op->dL_db[i] = op->dL_db[i] * momentum + (1 - momentum) * op->dL_db_now[i];
				}
				APPLY_GRADIENTS(op->dL_db, op->bias, lr, op->bias_size);
				APPLY_GRADIENTS(op->dL_dw, op->weights, lr, op->weights_size);
			}
			else
			{
				for (uint32_t i = 0; i < op->weights_size; i++)
				{
					op->dL_dw[i] = op->dL_dw[i] * momentum + (1 - momentum) * op->dL_dw_now[i];
				}
				APPLY_GRADIENTS(op->dL_dw, op->weights, lr, op->weights_size);
			}
		}

		RMSProp::RMSProp(DTYPE lr, DTYPE beta) : beta(beta)
		{
			this->lr = lr;
		}

		void RMSProp::apply_gradients(Operator* op)
		{
			if (!op->dL_dw)
				op->dL_dw = new DTYPE[op->weights_size]{};
			if (!op->dL_dw_2)
				op->dL_dw_2 = new DTYPE[op->weights_size]{};
			if (op->bias)
			{
				if (!op->dL_db)
					op->dL_db = new DTYPE[op->bias_size]{};
				if (!op->dL_db_2)
					op->dL_db_2 = new DTYPE[op->bias_size]{};
				for (uint32_t i = 0; i < op->weights_size; i++)
				{
					op->dL_dw_2[i] = op->dL_dw_2[i] * beta + (1 - beta) * pow(op->dL_dw_now[i], 2);
					op->dL_dw[i] = op->dL_dw_now[i] / (sqrt(op->dL_dw_2[i]) + eps);
					if (i<op->bias_size)
					{
						op->dL_db_2[i] = op->dL_db_2[i] * beta + (1 - beta) * pow(op->dL_db_now[i], 2);
						op->dL_db[i] = op->dL_db_now[i] / (sqrt(op->dL_db_2[i]) + eps);
					}
				}
				APPLY_GRADIENTS(op->dL_db, op->bias, lr, op->bias_size);
				APPLY_GRADIENTS(op->dL_dw, op->weights, lr, op->weights_size);
			}
			else
			{
				for (uint32_t i = 0; i < op->weights_size; i++)
				{
					op->dL_dw_2[i] = op->dL_dw_2[i] * beta + (1 - beta) * pow(op->dL_dw_now[i], 2);
					op->dL_dw[i] = op->dL_dw_now[i] / (sqrt(op->dL_dw_2[i]) + eps);
				}
				APPLY_GRADIENTS(op->dL_dw, op->weights, lr, op->weights_size);
			}
		}

		Adam::Adam(DTYPE lr, DTYPE beta1, DTYPE beta2) :beta1(beta1), beta2(beta2)
		{
			this->lr = lr;
		}

		void Adam::apply_gradients(Operator* op)
		{
			if (!op->dL_dw)
				op->dL_dw = new DTYPE[op->weights_size]{};
			if (!op->dL_dw_1)
				op->dL_dw_1 = new DTYPE[op->weights_size]{};
			if (!op->dL_dw_2)
				op->dL_dw_2 = new DTYPE[op->weights_size]{};
			time_step += 1;
			if (op->bias)
			{
				if (!op->dL_db)
					op->dL_db = new DTYPE[op->bias_size]{};
				if (!op->dL_db_1)
					op->dL_db_1 = new DTYPE[op->bias_size]{};
				if (!op->dL_db_2)
					op->dL_db_2 = new DTYPE[op->bias_size]{};
				for (uint32_t i = 0; i < op->weights_size; i++)
				{
					op->dL_dw_2[i] = op->dL_dw_2[i] * beta2 + (1 - beta2) * pow(op->dL_dw_now[i], 2);
					op->dL_dw_1[i] = op->dL_dw_1[i] * beta1 + (1 - beta1) * op->dL_dw_now[i];
					op->dL_dw[i] = op->dL_dw_1[i] / (1 - pow(beta1, time_step)) / (sqrt(op->dL_dw_2[i] / (1 - pow(beta2, time_step))) + eps);
					if (i < op->bias_size)
					{
						op->dL_db_2[i] = op->dL_db_2[i] * beta2 + (1 - beta2) * pow(op->dL_db_now[i], 2);
						op->dL_db_1[i] = op->dL_db_1[i] * beta1 + (1 - beta1) * op->dL_db_now[i];
						op->dL_db[i] = op->dL_db_1[i] / (1 - pow(beta1, time_step)) / (sqrt(op->dL_db_2[i] / (1 - pow(beta2, time_step))) + eps);
					}
				}
				APPLY_GRADIENTS(op->dL_db, op->bias, lr, op->bias_size);
				APPLY_GRADIENTS(op->dL_dw, op->weights, lr, op->weights_size);
			}
			else
			{
				for (uint32_t i = 0; i < op->weights_size; i++)
				{
					op->dL_dw_2[i] = op->dL_dw_2[i] * beta2 + (1 - beta2) * pow(op->dL_dw_now[i], 2);
					op->dL_dw_1[i] = op->dL_dw_1[i] * beta1 + (1 - beta1) * op->dL_dw_now[i];
					op->dL_dw[i] = op->dL_dw_1[i] / (1 - pow(beta1, time_step)) / (sqrt(op->dL_dw_2[i] / (1 - pow(beta2, time_step))) + eps);
				}
				APPLY_GRADIENTS(op->dL_dw, op->weights, lr, op->weights_size);
			}
		}
	}
}
