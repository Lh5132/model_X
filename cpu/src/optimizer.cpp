#include "optimizer.h"
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

		void SGD::apply_gradients(Dense * dense)
		{
			for (uint32_t i = 0; i < dense->out_size; i++)
			{
				DTYPE* w_grad = dense->dL_dw_now + i * dense->in_size;
				DTYPE* w_data = dense->weights + i * dense->in_size;
				if (dense->with_bias)
					dense->bias[i] = dense->bias[i] - lr * dense->dL_db_now[i];
				for (uint32_t j = 0; j < dense->in_size; j++)
				{
					w_data[j] = w_data[i] - lr * w_grad[j];
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
				uint32_t w_channel_loc = c * conv->kernel_steps;
				DTYPE* w_data = conv->get_channel_data(c);
				DTYPE* w_grad = conv->dL_dw + w_channel_loc;
				DTYPE* w_grad_now = conv->dL_dw_now + w_channel_loc;
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

		void Momentum::apply_gradients(Dense * dense)
		{
			if (!dense->dL_dw)
				dense->dL_dw = new DTYPE[dense->in_size * dense->out_size]{};
			if (dense->with_bias)
			{
				if (!dense->dL_db)
					dense->dL_db = new DTYPE[dense->out_size];
			}
			for (uint32_t i = 0; i < dense->out_size; i++)
			{
				uint32_t w_loc = i * dense->in_size;
				DTYPE* w_data = dense->weights + i*dense->in_size;
				DTYPE* w_grad = dense->dL_dw + w_loc;
				DTYPE* w_grad_now = dense->dL_dw_now + w_loc;
				if (dense->with_bias)
				{
					dense->dL_db[i] = dense->dL_db[i] * momentum + (1 - momentum)*dense->dL_db_now[i];
					dense->bias[i] = dense->bias[i] - lr * dense->dL_db[i];
				}
				for (uint32_t j = 0; j < dense->in_size; j++)
				{
					w_grad[j] = w_grad[j] * momentum + (1 - momentum)*w_grad_now[j];
					w_data[j] = w_data[j] - lr * w_grad[j];
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
				uint32_t w_loc = c * conv->kernel_steps;
				DTYPE* w_data = conv->get_channel_data(c);
				DTYPE* w_grad = conv->dL_dw + w_loc;
				DTYPE* w_grad_2 = conv->dL_dw_2 + w_loc;
				DTYPE* w_grad_now = conv->dL_dw_now + w_loc;
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

		void RMSProp::apply_gradients(Dense * dense)
		{
			if (!dense->dL_dw)
				dense->dL_dw = new DTYPE[dense->in_size * dense->out_size]{};
			if (!dense->dL_dw_2)
				dense->dL_dw_2 = new DTYPE[dense->in_size * dense->out_size]{};
			if (dense->with_bias)
			{
				if (!dense->dL_db)
					dense->dL_db = new DTYPE[dense->out_size]{};
				if (!dense->dL_db_2)
					dense->dL_db_2 = new DTYPE[dense->out_size]{};
			}
			for (uint32_t i = 0; i < dense->out_size; i++)
			{
				uint32_t w_loc = i * dense->in_size;
				DTYPE* w_data = dense->weights + i*dense->in_size;
				DTYPE* w_grad = dense->dL_dw + w_loc;
				DTYPE* w_grad_2 = dense->dL_dw_2 + w_loc;
				DTYPE* w_grad_now = dense->dL_dw_now + w_loc;
				if (dense->with_bias)
				{
					dense->dL_db_2[i] = dense->dL_db_2[i] * beta + (1 - beta)*pow(dense->dL_db_now[i], 2);
					dense->dL_db[i] = dense->dL_db_now[i] / (sqrt(dense->dL_db_2[i]) + eps);
					dense->bias[i] = dense->bias[i] - lr * dense->dL_db[i];
				}
				for (uint32_t j = 0; j < dense->in_size; j++)
				{
					w_grad_2[j] = w_grad_2[j] * beta + (1 - beta)*pow(w_grad_now[j], 2);
					w_grad[j] = w_grad_now[j] / (sqrt(w_grad_2[j]) + eps);
					w_data[j] = w_data[j] - lr * w_grad[j];
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
			time_step += 1;
			for (uint16_t c = 0; c < conv->out_channels; c++)
			{
				uint32_t w_loc = c * conv->kernel_steps;
				DTYPE* w_data = conv->get_channel_data(c);
				DTYPE* w_grad = conv->dL_dw + w_loc;
				DTYPE* w_grad_1 = conv->dL_dw_1 + w_loc;
				DTYPE* w_grad_2 = conv->dL_dw_2 + w_loc;
				DTYPE* w_grad_now = conv->dL_dw_now + w_loc;
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
					w_data[c] = w_data[c] - lr * w_grad[c];
				}
			}
		}
		void Adam::apply_gradients(Dense* dense)
		{
			if (!dense->dL_dw)
				dense->dL_dw = new DTYPE[dense->in_size * dense->out_size]{};
			if (!dense->dL_dw_1)
				dense->dL_dw_1 = new DTYPE[dense->in_size * dense->out_size]{};
			if (!dense->dL_dw_2)
				dense->dL_dw_2 = new DTYPE[dense->in_size * dense->out_size]{};
			if (dense->with_bias)
			{
				if (!dense->dL_db)
					dense->dL_db = new DTYPE[dense->in_size]{};
				if (!dense->dL_db_1)
					dense->dL_db_1 = new DTYPE[dense->in_size]{};
				if (!dense->dL_db_2)
					dense->dL_db_2 = new DTYPE[dense->in_size]{};
			}
			time_step += 1;
			for (uint32_t i = 0; i < dense->out_size; i++)
			{
				uint32_t w_loc = i * dense->in_size;
				DTYPE* w_data = dense->weights + i * dense->in_size;
				DTYPE* w_grad = dense->dL_dw + w_loc;
				DTYPE* w_grad_1 = dense->dL_dw_1 + w_loc;
				DTYPE* w_grad_2 = dense->dL_dw_2 + w_loc;
				DTYPE* w_grad_now = dense->dL_dw_now + w_loc;
				if (dense->with_bias)
				{
					dense->dL_db_2[i] = dense->dL_db_2[i] * beta2 + (1 - beta2)*pow(dense->dL_db_now[i], 2);
					dense->dL_db_1[i] = dense->dL_db_1[i] * beta1 + (1 - beta1)*dense->dL_db_now[i];
					dense->dL_db[i] = dense->dL_db_1[i] / (1 - pow(beta1, time_step)) / (sqrt(dense->dL_db_2[i] / (1 - pow(beta2, time_step))) + eps);
					dense->bias[i] = dense->bias[i] - lr * dense->dL_db[i];
				}
				for (uint32_t j = 0; j < dense->in_size; j++)
				{
					w_grad_2[j] = w_grad_2[j] * beta2 + (1 - beta2)*pow(w_grad_now[j], 2);
					w_grad_1[j] = w_grad_1[j] * beta1 + (1 - beta1)*w_grad_now[j];
					w_grad[j] = w_grad_1[j] / (1 - pow(beta1, time_step)) / (sqrt(w_grad_2[j] / (1 - pow(beta2, time_step))) + eps);
					w_data[j] = w_data[j] - lr * w_grad[j];
				}
			}
		}
	}

}
