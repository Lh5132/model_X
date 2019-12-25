#include "model.h"
#include "storage.h"
#include <iostream>
#include <ctime>
#include "optimizer.h"

using namespace model_X;
using namespace std;


void test_trans()
{
	storage n1 = storage_creater::creat(2,3,8,8);
	n1->random_init();
	cout << "原始矩阵: " << endl;
	n1->print_data();

	Conv_2d conv(3, 2, 3, 3, conv_stride(1,1), conv_padding(PADDING_STYLE::	SAME));
	conv.random_init();
	conv.set_async_thread(4);
	conv.print_weight();
	storage n2 = conv.forward(n1);
	cout << "输出" << endl;
	n2->print_data();


	Batch_normal_2d BN = Batch_normal_2d(2);
	Max_pool mp = Max_pool(2,2);
	BN.train();
	Relu relu = Relu();
	Drop_out dp = Drop_out();
	storage n5 = dp.forward(n2);
	cout << "Drop_out输出" << endl;
	n5->print_data();

	Dense dense(n5->batch_steps, 8, true);
	dense.set_async_thread(4);
	dense.random_init();

	storage n3 = dense.forward(n5);
	cout << "全连接输出" << endl;
	n3->print_data();

	Soft_max sm = Soft_max();
	storage n4 = sm.forward(n3);
	cout << "softmax输出: " << endl;
	n4->print_data();
	getchar();
}

void test_speed()
{
	storage n1 = storage_creater::creat(1, 3, 224, 224);
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
	storage n2;
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

	storage input = storage_creater::creat(1, 3, 224, 224);
	input->random_init();
	model.eval();
	model.set_async_thread();
	cout << "模型创建完成" << endl;
	long start = clock();
	storage out = model.forward(input);
	long end = clock();
	cout << "耗时: " << end - start << endl;
	out->print_shape();
	cout << endl;
	getchar();
}
void test_conv_backward()
{
	storage n1 = storage_creater::creat(1, 3, 5, 5);
	float inp[] = { 0.8398, 0.0308, 0.0695, 0.6658, 0.3044, 0.4174, 0.5668, 0.7305, 0.2353,
		0.4678, 0.9259, 0.6297, 0.0365, 0.3951, 0.5539, 0.6972, 0.5534, 0.1443,
		0.5309, 0.1678, 0.4845, 0.6227, 0.6950, 0.9202, 0.6015, 0.6132, 0.7825,
		0.8218, 0.5978, 0.9807, 0.7549, 0.3665, 0.8232, 0.3721, 0.4600, 0.6462,
		0.1541, 0.2583, 0.4188, 0.0606, 0.6783, 0.3117, 0.7399, 0.9236, 0.1688,
		0.6016, 0.4300, 0.5211, 0.2801, 0.3428, 0.9276, 0.5343, 0.6967, 0.0819,
		0.5186, 0.1484, 0.2159, 0.7510, 0.6419, 0.4928, 0.6240, 0.7414, 0.2302,
		0.6409, 0.6579, 0.6359, 0.7092, 0.5766, 0.4834, 0.6631, 0.1424, 0.0218,
		0.7714, 0.9141, 0.9634 };
	memcpy(n1->get_batch_data(0), inp, 75 * sizeof(float));
	n1->print_data();
	Conv_2d conv(3, 2, 3, 3, conv_stride(1, 1), conv_padding(PADDING_STYLE::SAME));
	float w1[] = { -0.0949, -0.0137,  0.1447,  0.1564,  0.1240, -0.0569, -0.1180,  0.0368,
		0.1126, -0.1686, -0.1784, -0.1684,  0.0618, -0.0641, -0.1098,  0.1559,
		0.0911, -0.0465, -0.0648,  0.1606, -0.1355, -0.0502,  0.0205,  0.1804,
		-0.0304,  0.0491, -0.0552 };
	float w2[] = { -0.1661,  0.0395,  0.1818,  0.0101, -0.0310,  0.1640, -0.0512, -0.1813,
		0.0333,  0.1029, -0.0351,  0.0250, -0.0777, -0.0914,  0.1260, -0.1758,
		-0.1844, -0.1249,  0.0299, -0.0903,  0.1078,  0.0378,  0.1167, -0.1790,
		-0.1239, -0.1832, -0.1589 };

	memcpy(conv.get_channel_data(0), w1, 27 * sizeof(float));
	memcpy(conv.get_channel_data(1), w2, 27 * sizeof(float));

	conv.print_weight();
	n1->require_grad();
	conv.train();
	storage n2 = conv.forward(n1);
	n2->print_data();
	conv.dL_dout = new storage(1,2,5,5,true);
	conv.dL_dout->set_one();
	auto opt = Optimizer::SGD(0.1);
	conv.backward(opt);
	conv.dL_din->print_data();
	getchar();
}
int main(int arc, char** argv)
{
	test_conv_backward();
}