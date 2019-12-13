#include "model.h"
#include "node.h"
#include <iostream>
#include <ctime>

using namespace model_X;
using namespace std;


void test_trans()
{
	Node n1 = Node_creater::creat(2,3,8,8);
	n1->random_init();
	cout << "原始矩阵: " << endl;
	n1->print_data();

	Conv_2d conv(3, 2, 3, 3, conv_stride(1,1), conv_padding(PADDING_STYLE::	SAME));
	conv.random_init();
	conv.set_async_thread(4);
	conv.print_weight();
	Node n2 = conv.forward(n1);
	cout << "输出" << endl;
	n2->print_data();


	Batch_normal_2d BN = Batch_normal_2d(2);
	Max_pool mp = Max_pool(2,2);
	BN.train();
	Relu relu = Relu();
	Drop_out dp = Drop_out();
	Node n5 = dp.forward(n2);
	cout << "Drop_out输出" << endl;
	n5->print_data();

	Dense dense(n5->batch_steps, 8, true);
	dense.set_async_thread(4);
	dense.random_init();

	Node n3 = dense.forward(n5);
	cout << "全连接输出" << endl;
	n3->print_data();

	Soft_max sm = Soft_max();
	Node n4 = sm.forward(n3);
	cout << "softmax输出: " << endl;
	n4->print_data();
	getchar();
}

void test_speed()
{
	Node n1 = Node_creater::creat(1, 3, 224, 224);
	n1->random_init();
	Conv_2d conv(3, 64, 3, 3, conv_stride(1, 1), conv_padding(PADDING_STYLE::SAME));
	Relu relu = Relu();
	Max_pool mp = Max_pool(2, 2);
	Batch_normal_2d BN = Batch_normal_2d(512);
	Drop_out dp = Drop_out(0.5);
	Dense dn = Dense(n1->batch_steps, 1000);
	dn.set_async_thread();
	conv.random_init();
	conv.set_async_thread(8);
	Node n2;
	long t1 = clock();
	n2 = conv.forward(n1);
	long t2 = clock();
	cout << "conv: " << t2 - t1 << endl;
	t1 = clock();
	n2 = BN.forward(n2);
	t2 = clock();
	cout << "BN: " << t2 - t1 << endl;
	t1 = clock();
	n2 = dp.forward(n2);
	t2 = clock();
	cout << "dp: " << t2 - t1 << endl;
	t1 = clock();
	n2 = relu.forward(n2);
	t2 = clock();
	cout << "relu: " << t2 - t1 << endl;
	t1 = clock();
	n2 = mp.forward(n2);
	t2 = clock();
	cout << "max_pool: " << t2 - t1 << endl;
	t1 = clock();
	n2 = dn.forward(n1);
	t2 = clock();
	cout << "dense: " << t2 - t1 << endl;
	getchar();
}

void test_model()
{
	Sequential model = Sequential();
	model.add_moudle(new Conv_2d(3, 64, 3, 3, conv_stride(1,1), conv_padding(1,1,1,1)));
	model.add_moudle(new Relu());
	model.add_moudle(new Conv_2d(64, 64, 3, 3, conv_stride(1, 1), conv_padding(1, 1, 1, 1)));
	model.add_moudle(new Relu());
	model.add_moudle(new Max_pool(2, 2));

	model.add_moudle(new Conv_2d(64, 128, 3, 3, conv_stride(1, 1), conv_padding(1, 1, 1, 1)));
	model.add_moudle(new Relu());
	model.add_moudle(new Conv_2d(128, 128, 3, 3, conv_stride(1, 1), conv_padding(1, 1, 1, 1)));
	model.add_moudle(new Relu());
	model.add_moudle(new Max_pool(2, 2));

	model.add_moudle(new Conv_2d(128, 256, 3, 3, conv_stride(1, 1), conv_padding(1, 1, 1, 1)));
	model.add_moudle(new Relu());
	model.add_moudle(new Conv_2d(256, 256, 3, 3, conv_stride(1, 1), conv_padding(1, 1, 1, 1)));
	model.add_moudle(new Relu());
	model.add_moudle(new Conv_2d(256, 256, 3, 3, conv_stride(1, 1), conv_padding(1, 1, 1, 1)));
	model.add_moudle(new Relu());
	model.add_moudle(new Max_pool(2, 2));

	model.add_moudle(new Conv_2d(256, 512, 3, 3, conv_stride(1, 1), conv_padding(1, 1, 1, 1)));
	model.add_moudle(new Relu());
	model.add_moudle(new Conv_2d(512, 512, 3, 3, conv_stride(1, 1), conv_padding(1, 1, 1, 1)));
	model.add_moudle(new Relu());
	model.add_moudle(new Conv_2d(512, 512, 3, 3, conv_stride(1, 1), conv_padding(1, 1, 1, 1)));
	model.add_moudle(new Relu());
	model.add_moudle(new Max_pool(2, 2));

	model.add_moudle(new Conv_2d(512, 512, 3, 3, conv_stride(1, 1), conv_padding(1, 1, 1, 1)));
	model.add_moudle(new Relu());
	model.add_moudle(new Conv_2d(512, 512, 3, 3, conv_stride(1, 1), conv_padding(1, 1, 1, 1)));
	model.add_moudle(new Relu());
	model.add_moudle(new Conv_2d(512, 512, 3, 3, conv_stride(1, 1), conv_padding(1, 1, 1, 1)));
	model.add_moudle(new Relu());
	model.add_moudle(new Max_pool(2, 2));

	model.add_moudle(new Dense(25088, 4096));
	model.add_moudle(new Relu());
	model.add_moudle(new Drop_out());
	model.add_moudle(new Dense(4096, 4096));
	model.add_moudle(new Relu());
	model.add_moudle(new Drop_out());
	model.add_moudle(new Dense(4096, 1000));

	Node input = Node_creater::creat(1, 3, 224, 224);
	input->random_init();
	model.eval();
	model.set_async_thread();
	cout << "模型创建完成" << endl;
	long start = clock();
	Node out = model.forward(input);
	long end = clock();
	cout << "耗时: " << end - start << endl;
	out->print_shape();
	cout << endl;
	getchar();
}
void test_conv_backend()
{
	Node n1 = Node_creater::creat(1, 3, 5, 5);
	n1->random_init();
	n1->print_data();
	Conv_2d conv(3, 2, 3, 3, conv_stride(1, 1), conv_padding(PADDING_STYLE::SAME));
	conv.random_init();
	n1->require_grad();
	conv.train();
	Node n2 = conv.forward(n1);
	n2->print_data();
	conv.dL_dout = new node(1,2,5,5,true);
	conv.dL_dout->set_one();
	Optimizer opt = Optimizer(Optimizer_method::SGD,0.1);
	conv.backend(opt);
	conv.dL_din->print_data();
	getchar();
}
int main(int arc, char** argv)
{
	test_conv_backend();
}