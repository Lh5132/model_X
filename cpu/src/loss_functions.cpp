#include "loss_functions.h"
#include <vector>
#include <unordered_set>
namespace model_X
{
	Loss::Loss(Operator* backward_start, DTYPE loss) :backward_start(backward_start), loss(loss) {}
	Loss::Loss(const Loss& other): backward_start(other.backward_start), loss(other.loss) {}
	DTYPE Loss::item()
	{
		return this->loss;
	}
	/*
	求导思路如下：
		初始化一个stack和set，stack中装入最后一个Operator作为求导的起点，set则为空用于放入有分支的Operator；
		当向前求导的过程中遇到Concator时，则把Concator链接的两个节点放入stack中
		当向前求导的过程中遇到某一个Operator的count_out>1时,说明该节点为某一条分支的起始点，进行如下处理：
			首先判断该Operator是否存在于set中，若set中没有，说明是第一次遇到该Operator，将其放入set中，并从
			stack中取出下一个Operator进行求导
			若set中已存在Operator，但是其count_back<count_out,说明该节点还有分支未进行计算，因此停止求导，并
			从stack中取出下一个Operator进行求导
			若set中已存在Operator,并且其count_back=count_out，说明该节点所有分支均已计算完毕，将其从set中移除
			并将该节点放入stack中
		当求导进行到链表起点，或者stack中没有Operator时，求导过程结束
	*/
	void Loss::backward(Optimizer::base_optimizer& opt)
	{
		Operator* now;
		vector<Operator*> stack = { this->backward_start };
		unordered_set<Operator*> set;
		while (stack.size())
		{
			now = stack[stack.size() - 1];
			stack.pop_back();
			if (now->ID == LAYER_ID::CONCAT)
			{
				((Concator*)now)->set_gradients();
				stack.push_back(((Concator*)now)->get_O1());
				stack.push_back(((Concator*)now)->get_O2());
				continue;
			}
			else
			{
				while (now)
				{
					if (now->ID == LAYER_ID::CONCAT)
					{
						((Concator*)now)->set_gradients();
						stack.push_back(((Concator*)now)->get_O1());
						stack.push_back(((Concator*)now)->get_O2());
						break;
					}
					now->backward(opt);
					now->get_pre()->increase_count_back();
					now->pass_gradients();
					if (now->get_pre()->get_count_back() < now->get_pre()->get_count_out())
					{
						if (set.count(now->get_pre()) == 0)
						{
							set.insert(now->get_pre());
						}
						else if (now->get_pre()->get_count_back() == now->get_pre()->get_count_out())
						{
							set.erase(now->get_pre());
							stack.push_back(now->get_pre());
						}
						break;
					}
					else
						now = now->get_pre();
				}
			}
		}
	}
	namespace Loss_Functions
	{

		Loss BCELoss(const storage& output, const storage& ground_truth)
		{
			if (output->batch_steps > 1)
			{
				throw "BCELoss requires single output";
			}
			DTYPE out = 0;
			if(!output->creater->dL_dout)
				output->creater->dL_dout = output->copy(false);
			for (uint16_t c = 0; c < output->batchsize; c++)
			{
				DTYPE exp_ = exp(0 - output->get_batch_data(c)[0]);
				DTYPE predict = 1 / (1 + exp_);
				DTYPE* gradient_batch = output->creater->dL_dout->get_batch_data(c);
				if (ground_truth->get_batch_data(c)[0] == 0.0f)
				{
					out -= log(1 - predict);
					*gradient_batch = predict / output->batchsize;
				}
				else
				{
					out -= log(predict);
					*gradient_batch = (predict -1) / output->batchsize;
				}
			}
			Loss l(output->creater, out / output->batchsize);
			return l;
		}
		Loss MSELoss(const storage& output, const storage& ground_truth)
		{
			if (output->batch_steps > 1)
			{
				throw "BCELoss requires single output";
			}
			DTYPE out = 0;
			output->creater->dL_dout = output->copy(false);
			for (uint16_t c = 0; c < output->batchsize; c++)
			{
				DTYPE* gradient_batch = output->creater->dL_dout->get_batch_data(c);
				out += pow((output->get_batch_data(c)[0] - ground_truth->get_batch_data(c)[0]), 2) / 2;
				*gradient_batch = (output->get_batch_data(c)[0] - ground_truth->get_batch_data(c)[0]) / output->batchsize;
			}
			Loss l(output->creater, out / output->batchsize);
			return l;
		}
		Loss SOFTMAXLoss(const storage& output, const storage& ground_truth)
		{
			if (output->batch_steps == 1)
			{
				throw "SOFTMAXLoss requires muti output";
			}
			DTYPE out = 0;
			output->creater->dL_dout = output->copy(false);
			DTYPE* predict = new DTYPE[output->batchsize*output->batch_steps]{};
			DTYPE* exp_ = new DTYPE[output->batchsize*output->batch_steps]{};
			for (uint16_t c = 0; c < output->batchsize; c++)
			{
				uint32_t pad_loc = c*output->batch_steps_pad;
				uint32_t loc = c*output->batch_steps;
				DTYPE* out_put_data = output->data + pad_loc;
				DTYPE* ground_truth_data = ground_truth->data + pad_loc;
				DTYPE batch_loss = 0;
				DTYPE* gradient_batch = output->creater->dL_dout->data + pad_loc;
				DTYPE* predict_batch = predict + loc;
				DTYPE total = 0;
				DTYPE* exp_batch = exp_ + loc;
				for (uint16_t i = 0; i < output->batch_steps; i++)
				{
					exp_[i] = exp(out_put_data[i]);
					total += exp_[i];
				}
				for (uint16_t i = 0; i < output->batch_steps; i++)
				{
					predict_batch[i] = exp_[i]/total;
					if (ground_truth_data[i] == 0.0f)
					{
						batch_loss -= log(1 - predict_batch[i]);
						gradient_batch[i] = exp_batch[i]/total / output->batchsize;
					}
					else
					{
						batch_loss -= log(predict_batch[i]);
						gradient_batch[i] = (exp_batch[i] - total) / total / output->batchsize;
					}
				}
				out += batch_loss;
			}
			Loss l(output->creater, out / output->batchsize);
			return l;
		}
		Loss BCEWithLogitsLoss(const storage& output, const storage& ground_truth)
		{
			if (output->batch_steps > 1)
			{
				throw "BCELoss requires single output";
			}
			DTYPE out = 0;
			output->creater->dL_dout = output->copy(false);
			for (uint16_t c = 0; c < output->batchsize; c++)
			{
				DTYPE predict = output->get_batch_data(c)[0];
				DTYPE* gradient_batch = output->creater->dL_dout->get_batch_data(c);
				if (ground_truth->get_batch_data(c)[0] == 0.0f)
				{
					out -= log(1 - predict);
					*gradient_batch = 1 / (1 - predict) / output->batchsize;
				}
				else
				{
					out -= log(predict);
					*gradient_batch = 0 - 1/predict / output->batchsize;
				}
			}
			Loss l(output->creater, out / output->batchsize);
			return l;
		}
	}
}