#include "model.h"
#include <iostream>

namespace model_X
{
	Operator* Base_Model::creat_moudle(Operator* op)
	{
		this->operators.push_back(op);
		return op;
	}
	Sequential::Sequential()
	{
	}
	Sequential::~Sequential()
	{
		for (Operator* layer : this->operators)
		{
			if (layer)
			{
				delete layer;
				layer = nullptr;
			}
		}
	}
	void Base_Model::random_init(int init_method)
	{
		for (Operator* layer : this->operators)
		{
			layer->random_init(init_method);
		}
	}
	void Base_Model::save(const char* path)
	{
		ofstream outfile(path, ios::binary);
		for (Operator* layer : this->operators)
		{
			layer->to_binay_file(outfile);
		}
		outfile.close();
	}
	void Base_Model::load(const char * path)
	{
		ifstream infile(path, ios::binary);
		char* id = new char;
		while (infile.read(id, 1))
		{
			switch ((*id))
			{
			case LAYER_ID::CONV_2D:
			{
				Conv_2d* conv = new Conv_2d();
				conv->read_stream(infile);
				this->operators.push_back(conv);
				break;
			}
			case LAYER_ID::DENSE:
			{
				Dense* dense = new Dense();
				dense->read_stream(infile);
				this->operators.push_back(dense);
				break;
			}
			case LAYER_ID::BN:
			{
				Batch_normal_2d* bn = new Batch_normal_2d();
				bn->read_stream(infile);
				this->operators.push_back(bn);
				break;
			}
			case LAYER_ID::RELU:
			{
				Relu* relu = new Relu();
				relu->read_stream(infile);
				this->operators.push_back(relu);
				break;
			}
			case LAYER_ID::MAX_POOL:
			{
				Max_pool* mp = new Max_pool();
				mp->read_stream(infile);
				this->operators.push_back(mp);
				break;
			}
			case LAYER_ID::AVE_POOL:
			{
				Ave_pool* ap = new Ave_pool();
				ap->read_stream(infile);
				this->operators.push_back(ap);
				break;
			}
			case LAYER_ID::SIGMOID:
			{
				Sigmoid* sig = new Sigmoid();
				sig->read_stream(infile);
				this->operators.push_back(sig);
				break;
			}
			case LAYER_ID::SOFT_MAX:
			{
				Soft_max* som = new Soft_max();
				som->read_stream(infile);
				this->operators.push_back(som);
				break;
			}
			case LAYER_ID::DROP_OUT:
			{
				Drop_out* d = new Drop_out();
				d->read_stream(infile);
				this->operators.push_back(d);
				break;
			}
			case LAYER_ID::CONCAT:
				break;
			default:
			{
				delete id;
				throw "weights file broken";
				break;
			}
			}
		}
		infile.close();
		delete id;
	}
	void Base_Model::eval()
	{
		for (Operator* layer : this->operators)
			layer->eval();
	}

	void Base_Model::train()
	{
		for (Operator* layer : this->operators)
			layer->train();
	}

	void Base_Model::set_async_thread()
	{
		for (Operator* layer : this->operators)
			layer->set_async_thread(cpu_cors);
	}

	void Base_Model::set_async_thread(int n)
	{
		for (Operator* layer : this->operators)
			layer->set_async_thread(n);
	}
	void Sequential::print_info()
	{
		for (int i = 0; i < this->operators.size(); i++)
		{
			cout << "layer: " << i + 1 << endl;
			printf("%s\n", this->operators[i]->info().c_str());
		}
	}
	void Sequential::add_moudle(Operator * layer)
	{
		this->operators.push_back(layer);
	}

	tensor Sequential::forward(tensor& input)
	{
		for (Operator* layer : this->operators)
		{
			input = layer->forward(input);
		}
		return input;
	}
}


