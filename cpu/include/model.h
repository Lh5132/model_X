﻿#pragma once
#include "layer.h"
#include "storage.h"
#include <vector>

using namespace std;
namespace model_X
{

	class Base_Model
	{
	protected:
		vector<Operator*> operators;
		Operator* creat_moudle(Operator* op);
	public:
		void random_init(int init_method = Normal);
		void set_async_thread();
		void set_async_thread(int n);
		void save(const char* path);
		void load(const char* path);
		void eval();
		void train();
	};




	class Sequential final : public Base_Model
	{	
	public:
		Sequential();
		~Sequential();
		void print_info();
		void add_moudle(Operator* layer);
		tensor forward(tensor& input);
	};
}
