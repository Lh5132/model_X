#pragma once
#include "layer.h"
#include "node.h"
#include <vector>

using namespace std;
namespace model_X
{

	class Base_Model:public Operator
	{
	protected:
		vector<Operator*> operators;
		Operator* creat_moudle(Operator* op);
	public:
		virtual Node forward(Node input) = 0;
	};




	class Sequential
	{
	public:
		vector<Operator*> layers;
		
		
		Sequential();
		~Sequential();

		void random_init(int init_method = Normal);
		void save(const char* path);
		void load(const char* path);
		void print_info();
		

		void add_moudle(Operator* layer);

		void eval();
		void train();

		void set_async_thread();
		void set_async_thread(int n);

		Node forward(Node input);
	};
}
