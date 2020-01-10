#include "model.h"
#include "storage.h"
#include <iostream>
#include <ctime>
#include "optimizer.h"

using namespace model_X;
using namespace std;
void test_se()
{
	tensor input({ 1,3,8,8 });
	input->random_init();
	Conv_2d conv1(3, 2, 3, 3, conv_stride(1, 1), conv_padding(PADDING_STYLE::SAME));
	conv1.random_init();
	input = conv1.forward(input);
	cout << input->data_str() << endl;
	Batch_normal_2d bn1 = Batch_normal_2d(2);
	bn1.train();
	input = bn1.forward(input);
	cout << input->data_str() << endl;
	Relu re = Relu();
	input = re.forward(input);
	cout << input->data_str() << endl;
	Max_pool p = Max_pool();
	input = p.forward(input);
	cout << input->data_str() << endl;
}
void test_model()
{
	Sequential model = Sequential();
	model.add_moudle(new Conv_2d(3, 64, 3, 3, conv_stride(1,1), conv_padding(1,1,1,1)));
	model.add_moudle(new Batch_normal_2d(64));
	model.add_moudle(new Relu());
	model.add_moudle(new Conv_2d(64, 64, 3, 3, conv_stride(1, 1), conv_padding(1, 1, 1, 1)));
	model.add_moudle(new Batch_normal_2d(64));
	model.add_moudle(new Relu());
	model.add_moudle(new Max_pool(2, 2));

	model.add_moudle(new Conv_2d(64, 128, 3, 3, conv_stride(1, 1), conv_padding(1, 1, 1, 1)));
	model.add_moudle(new Batch_normal_2d(128));
	model.add_moudle(new Relu());
	model.add_moudle(new Conv_2d(128, 128, 3, 3, conv_stride(1, 1), conv_padding(1, 1, 1, 1)));
	model.add_moudle(new Batch_normal_2d(128));
	model.add_moudle(new Relu());
	model.add_moudle(new Max_pool(2, 2));

	model.add_moudle(new Conv_2d(128, 256, 3, 3, conv_stride(1, 1), conv_padding(1, 1, 1, 1)));
	model.add_moudle(new Batch_normal_2d(256));
	model.add_moudle(new Relu());
	model.add_moudle(new Conv_2d(256, 256, 3, 3, conv_stride(1, 1), conv_padding(1, 1, 1, 1)));
	model.add_moudle(new Batch_normal_2d(256));
	model.add_moudle(new Relu());
	model.add_moudle(new Conv_2d(256, 256, 3, 3, conv_stride(1, 1), conv_padding(1, 1, 1, 1)));
	model.add_moudle(new Batch_normal_2d(256));
	model.add_moudle(new Relu());
	model.add_moudle(new Max_pool(2, 2));

	model.add_moudle(new Conv_2d(256, 512, 3, 3, conv_stride(1, 1), conv_padding(1, 1, 1, 1)));
	model.add_moudle(new Batch_normal_2d(512));
	model.add_moudle(new Relu());
	model.add_moudle(new Conv_2d(512, 512, 3, 3, conv_stride(1, 1), conv_padding(1, 1, 1, 1)));
	model.add_moudle(new Batch_normal_2d(512));
	model.add_moudle(new Relu());
	model.add_moudle(new Conv_2d(512, 512, 3, 3, conv_stride(1, 1), conv_padding(1, 1, 1, 1)));
	model.add_moudle(new Batch_normal_2d(512));
	model.add_moudle(new Relu());
	model.add_moudle(new Max_pool(2, 2));

	model.add_moudle(new Conv_2d(512, 512, 3, 3, conv_stride(1, 1), conv_padding(1, 1, 1, 1)));
	model.add_moudle(new Batch_normal_2d(512));
	model.add_moudle(new Relu());
	model.add_moudle(new Conv_2d(512, 512, 3, 3, conv_stride(1, 1), conv_padding(1, 1, 1, 1)));
	model.add_moudle(new Batch_normal_2d(512));
	model.add_moudle(new Relu());
	model.add_moudle(new Conv_2d(512, 512, 3, 3, conv_stride(1, 1), conv_padding(1, 1, 1, 1)));
	model.add_moudle(new Batch_normal_2d(512));
	model.add_moudle(new Relu());
	model.add_moudle(new Max_pool(2, 2));

	model.add_moudle(new Conv_2d(512, 1, 3, 3, conv_stride(1, 1), conv_padding(1, 1, 1, 1)));

	//model.add_moudle(new Dense(25088, 4096));
	//model.add_moudle(new Relu());
	//model.add_moudle(new Drop_out());
	//model.add_moudle(new Dense(4096, 4096));
	//model.add_moudle(new Relu());
	//model.add_moudle(new Drop_out());
	//model.add_moudle(new Dense(4096, 1000));

	tensor input({ 1, 3, 224, 224 });
	input->random_init();
	cout << "张量初始化完成" << endl;
	model.train();
	model.random_init();
	//model.set_async_thread();
	cout << "模型创建完成" << endl;
	long start = clock();
	tensor out = model.forward(input);
	long end = clock();
	cout << "耗时: " << end - start << endl;
	cout << out->data_str() << endl;
	cout << endl;
}
void test_speed()
{
	tensor n1({ 5, 3, 224, 224 });
	n1->random_init();
	Conv_2d conv(3, 64, 3, 3, conv_stride(1, 1), conv_padding(PADDING_STYLE::SAME));
	conv.random_init();
	long start = clock();
	tensor n2 = conv.forward(n1);
	cout << clock() - start << endl;
}
void test_conv_backward()
{
	tensor n1({ 1, 3, 5, 5 });
	float inp[] = { 0.4505, 0.5432, 0.3538, 0.2683, 0.9527, 0.7076, 0.5039, 0.0413, 0.2213,
		 0.6359, 0.0243, 0.4561, 0.2465, 0.9162, 0.4251, 0.4000, 0.5490, 0.7518,
		 0.6632, 0.0949, 0.3027, 0.0211, 0.9481, 0.4599, 0.0878, 0.8645, 0.9962,
		 0.7921, 0.8872, 0.5810, 0.5058, 0.4611, 0.4693, 0.2406, 0.7789, 0.8154,
		 0.3768, 0.8028, 0.5361, 0.9896, 0.5685, 0.6981, 0.5756, 0.0677, 0.9244,
		 0.5574, 0.4666, 0.8971, 0.1526, 0.4587, 0.0577, 0.2414, 0.9990, 0.7603,
		 0.6704, 0.3985, 0.1825, 0.5981, 0.3576, 0.6132, 0.3267, 0.2906, 0.7497,
		 0.6642, 0.9261, 0.7905, 0.0202, 0.9172, 0.5581, 0.9457, 0.1529, 0.9926,
		 0.6390, 0.6889, 0.4041 };
	memcpy(n1->data, inp, 75 * sizeof(float));
	Conv_2d conv(3, 2, 3, 3, conv_stride(1, 1), conv_padding(PADDING_STYLE::SAME));
	float w1[] = { 0.0186, -0.0192, -0.1690, -0.0426,  0.1220, -0.0698,  0.0753, -0.0977,
		  0.1788, -0.0088,  0.1169, -0.0845, -0.0489,  0.0175,  0.1540, -0.1543,
		  0.1405,  0.1027,  0.0414,  0.1138, -0.1183,  0.0334, -0.1226, -0.1102,
		  0.0828,  0.1615, -0.0757 };
	float w2[] = { 0.0832, -0.0445, -0.1633,  0.1340,  0.0226,  0.0902,  0.0997,  0.0286,
		 -0.0935, -0.1384, -0.1208,  0.0169,  0.1702,  0.1334, -0.0225, -0.0970,
		 -0.0201, -0.0823,  0.0403, -0.1212,  0.0010,  0.0538,  0.1235,  0.1898,
		  0.0134,  0.1350,  0.0833 };

	float bias[] = { -0.1674,  0.1276 };

	memcpy(conv.get_bias(), bias, 2 * sizeof(float));
	memcpy(conv.get_channel_data(0), w1, 27 * sizeof(float));
	memcpy(conv.get_channel_data(1), w2, 27 * sizeof(float));

	conv.print_weight();
	conv.train();
	cout << n1->data_str() << endl;
	tensor n2 = conv.forward(n1);
	cout << n2->data_str() << endl;;
	conv.dL_dout = new storage({ 1,2,5,5 });
	conv.dL_dout->set_one();
	auto opt = Optimizer::SGD(0.1);
	conv.backward(opt);
	conv.print_weight();
	cout << conv.dL_din->data_str() << endl;

}
int main(int arc, char** argv)
{
	test_conv_backward();
}