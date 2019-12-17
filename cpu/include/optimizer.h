#pragma once
#include "util.h"
namespace model_X
{
	class Conv_2d;
	namespace Optimizer
	{
		class base_optimizer
		{
		protected:
			DTYPE lr;
		public:
			virtual void apply_gradients(Conv_2d* conv) = 0;
			virtual ~base_optimizer() = default;
		};
		class SGD final : public base_optimizer
		{
		public:
			SGD(DTYPE lr);
			void apply_gradients(Conv_2d* conv);
		};
		class Momentum final : public base_optimizer
		{
		protected:
			DTYPE momentum;
		public:
			Momentum(DTYPE lr, DTYPE momentum = 0.9);
			void apply_gradients(Conv_2d* conv);
		};

		class RMSProp final : public base_optimizer
		{
		private:
			DTYPE beta;
			DTYPE eps = 1e-8;
		public:
			RMSProp(DTYPE lr, DTYPE beta = 0.99);
			void apply_gradients(Conv_2d* conv);
		};

		class Adam final : public base_optimizer
		{
		private:
			DTYPE alpha;
			DTYPE beta1;
			DTYPE beta2;
			uint32_t time_step = 0;
			DTYPE eps = 1e-8;
		public:
			Adam(DTYPE lr, DTYPE beta1 = 0.9, DTYPE beta2 = 0.99);
			void apply_gradients(Conv_2d* conv);
		};
	}
}