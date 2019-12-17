#include "optimizer.h"
#include "layer.h"
namespace model_X
{
	namespace Optimizer
	{
		SGD::SGD(DTYPE lr)
		{
			this->lr = lr;
		}
		void SGD::apply_gradients(Conv_2d* conv)
		{
			for (uint16_t c = 0; c < conv->out_channels; c++)
			{
				DTYPE* w_data = conv->get_channel_data(c);
				DTYPE* w_grad = conv->dL_dw_now + c * conv->kernel_steps;
				if (conv->with_bias)
					conv->bias[c] = conv->bias[c] - lr * conv->dL_db_now[c];
				for (uint16_t i = 0; i < conv->kernel_steps; i++)
				{
					w_data[i] = w_data[i] - lr * w_grad[i];
				}
			}
		}

		Momentum::Momentum(DTYPE lr, DTYPE momentum) :momentum(momentum) {}
		void Momentum::apply_gradients(Conv_2d* conv)
		{
			if (!conv->dL_dw)
			{
				conv->dL_dw = new DTYPE[conv->kernel_steps * conv->out_channels]{};
			}
			if (conv->with_bias)
			{
				if (!conv->dL_db)
					conv->dL_db = new DTYPE[conv->out_channels];
			}
			for (uint16_t c = 0; c < conv->out_channels; c++)
			{
				DTYPE* w_data = conv->get_channel_data(c);
				DTYPE* w_grad = conv->dL_dw + c * conv->kernel_steps;
				DTYPE* w_grad_now = conv->dL_dw_now + c * conv->kernel_steps;
				if (conv->with_bias)
				{
					conv->dL_db[c] = conv->dL_db[c] * momentum + (1 - momentum)*conv->dL_db_now[c];
					conv->bias[c] = conv->bias[c] - lr * conv->dL_db[c];
				}
				for (uint16_t i = 0; i < conv->kernel_steps; i++)
				{
					w_grad[i] = w_grad[i] * momentum + (1 - momentum)*w_grad_now[i];
					w_data[i] = w_data[i] - lr * w_grad[i];
				}
			}
		}



		RMSProp::RMSProp(DTYPE lr, DTYPE beta) : beta(beta)
		{
			this->lr = lr;
		}

		void RMSProp::apply_gradients(Conv_2d * conv)
		{
			if (!conv->dL_dw)
				conv->dL_dw = new DTYPE[conv->kernel_steps * conv->out_channels]{};
			if (!conv->dL_dw_2)
				conv->dL_dw_2 = new DTYPE[conv->kernel_steps * conv->out_channels]{};
			if (conv->with_bias)
			{
				if (!conv->dL_db)
					conv->dL_db = new DTYPE[conv->out_channels]{};
				if (!conv->dL_db_2)
					conv->dL_db_2 = new DTYPE[conv->out_channels]{};
			}
			for (uint16_t c = 0; c < conv->out_channels; c++)
			{
				DTYPE* w_data = conv->get_channel_data(c);
				DTYPE* w_grad = conv->dL_dw + c * conv->kernel_steps;
				DTYPE* w_grad_2 = conv->dL_dw_2 + c * conv->kernel_steps;
				DTYPE* w_grad_now = conv->dL_dw_now + c * conv->kernel_steps;
				if (conv->with_bias)
				{
					conv->dL_db_2[c] = conv->dL_db_2[c] * beta + (1 - beta)*pow(conv->dL_db_now[c], 2);
					conv->dL_db[c] = conv->dL_db_now[c] / (sqrt(conv->dL_db_2[c]) + eps);
					conv->bias[c] = conv->bias[c] - lr * conv->dL_db[c];
				}
				for (uint16_t i = 0; i < conv->kernel_steps; i++)
				{
					w_grad_2[i] = w_grad_2[i] * beta + (1 - beta)*pow(w_grad_now[i], 2);
					w_grad[i] = w_grad_now[i] / (sqrt(w_grad_2[i]) + eps);
					w_data[i] = w_data[i] - lr * w_grad[i];
				}
			}
		}

		Adam::Adam(DTYPE lr, DTYPE beta1, DTYPE beta2) :beta1(beta1), beta2(beta2)
		{
			this->lr = lr;
		}

		void Adam::apply_gradients(Conv_2d * conv)
		{
			if (!conv->dL_dw)
				conv->dL_dw = new DTYPE[conv->kernel_steps * conv->out_channels]{};
			if (!conv->dL_dw_1)
				conv->dL_dw_1 = new DTYPE[conv->kernel_steps * conv->out_channels]{};
			if (!conv->dL_dw_2)
				conv->dL_dw_2 = new DTYPE[conv->kernel_steps * conv->out_channels]{};
			if (conv->with_bias)
			{
				if (!conv->dL_db)
					conv->dL_db = new DTYPE[conv->out_channels]{};
				if (!conv->dL_db_1)
					conv->dL_db_1 = new DTYPE[conv->out_channels]{};
				if (!conv->dL_db_2)
					conv->dL_db_2 = new DTYPE[conv->out_channels]{};
			}
			++time_step;
			for (uint16_t c = 0; c < conv->out_channels; c++)
			{
				DTYPE* w_data = conv->get_channel_data(c);
				DTYPE* w_grad = conv->dL_dw + c * conv->kernel_steps;
				DTYPE* w_grad_1 = conv->dL_dw_1 + c * conv->kernel_steps;
				DTYPE* w_grad_2 = conv->dL_dw_2 + c * conv->kernel_steps;
				DTYPE* w_grad_now = conv->dL_dw_now + c * conv->kernel_steps;
				if (conv->with_bias)
				{
					conv->dL_db_2[c] = conv->dL_db_2[c] * beta2 + (1 - beta2)*pow(conv->dL_db_now[c], 2);
					conv->dL_db_1[c] = conv->dL_db_1[c] * beta1 + (1 - beta1)*conv->dL_db_now[c];
					conv->dL_db[c] = conv->dL_db_1[c] / (1 - pow(beta1, time_step)) / (sqrt(conv->dL_db_2[c] / (1 - pow(beta2, time_step))) + eps);
					conv->bias[c] = conv->bias[c] - lr * conv->dL_db[c];
				}
				for (uint16_t i = 0; i < conv->kernel_steps; i++)
				{
					w_grad_2[c] = w_grad_2[c] * beta2 + (1 - beta2)*pow(w_grad_now[c], 2);
					w_grad_1[c] = w_grad_1[c] * beta1 + (1 - beta1)*w_grad_now[c];
					w_grad[c] = w_grad_1[c] / (1 - pow(beta1, time_step)) / (sqrt(w_grad_2[c] / (1 - pow(beta2, time_step))) + eps);
					conv->bias[c] = conv->bias[c] - lr * conv->dL_db[c];
				}
			}
		}
	}

}
