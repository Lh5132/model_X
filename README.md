# Model_X

* 利用纯c++代码构建的卷积神经网络训练及推理引擎，不需要依赖任何第三方库即可完成编译，支持AVX2加速与单卡GPU加速(可选)
* 支持的层操作包括Conv,BN,Dense,Drop,Relu,Maxpool,AvgPool,Sigmoid,Softmax
* 目前暂不支持上采样和反卷积
* 目前支持自定义计算图(包括残差链接，attention等结构)，但不支持自定义损失函数
* 梯度更新求解器包括SGD,Momentum,RMSProp,Adam四种
* 可通过model.set_async_thread()设置cpu下的并行计算，可自行填入线程数，默认值为电脑CPU总核心数
* 可通过model.to_cuda()和Node.to_cuda将模型和张量数据放入GPU中进行并行计算，计算结束后可通过model.to_cpu()和Node.to_cpu将数据拷贝至CPU中
* Python的API接口正在开发中

## 代码示例
### 1. 模型定义
#### 1.1 通过继承基类构建模型
````c
#include "Base_model.h"
#include "layer.h"
int main()
{
    //继承Base_model
    Class mymodel:public Base_model
    {
        //定义模型的层
        //定义时需采用creat_model()函数进行包裹
        Conv_2d conv1 = creat_model(new Conv_2d(3, 64, 3, 3, conv_stride(1,1), conv_padding(1,1,1,1)));
        Batch_normal_2d BN1 = creat_model(new Batch_normal_2d(64))
        Relu r1 = creat_model(new Relu());

        Conv_2d conv2 = creat_model(new Conv_2d(64, 128, 3, 3, conv_stride(1, 1), conv_padding(1, 1, 1, 1)));
        Batch_normal_2d BN2 = creat_model(new Batch_normal_2d(128))
        Relu r2 = creat_model(new Relu());

        Conv_2d conv3 = creat_model(new Conv_2d(64, 128, 3, 3, conv_stride(1, 1), conv_padding(1, 1, 1, 1)));
        Batch_normal_2d BN3 = creat_model(new Batch_normal_2d(128))
        Relu r3 = creat_model(new Relu());

        Drop_out dp = creat_model(new Drop_out());
        Dense dense = creat_model(new Dense(4096, 1000));

        //重载forward函数
        Node forward(Node input)
        {
            //自定义前向计算图
            Node x = conv1(input);
            x = BN1(x);
            x = r1(x);
            Node x1 = conv2(input);
            x1 = BN2(x1);
            x1 = r2(x1);
            Node x2 = conv3(input);
            x2 = BN2(x2);
            x2 = r2(x2);
            out = x1+x2;
            out = conv3(out);
            out = BN3(out);
            out = r3(out);
            out = dp(out);
            out = dense(out);
            return out;
        }
    }
}
````
#### 1.2 通过Sequential搭建模型
````c
int main()
{
    Sequential model = Sequential();
	model.add_moudle(new Conv_2d(3, 64, 3, 3, conv_stride(1,1), conv_padding(1,1,1,1)));
	model.add_moudle(new Relu());
	model.add_moudle(new Max_pool(2, 2));

	model.add_moudle(new Conv_2d(64, 128, 3, 3, conv_stride(1, 1), conv_padding(1, 1, 1, 1)));
	model.add_moudle(new Relu());
	model.add_moudle(new Max_pool(2, 2));

	model.add_moudle(new Dense(4096, 4096));
	model.add_moudle(new Relu());
	model.add_moudle(new Drop_out());
	model.add_moudle(new Dense(4096, 1000));
}
````
### 2. 模型训练
````c
int main()
{
    ...
    mymodel model = mymodel();
    model.train();
    Optimizer::Adam opt(0.1);
    for(uint32_t i=0;i<train_steps;i++)
    {
        //读取当前批次的数据及标签
        Node input = Input[i];
        Node ground_truch = Ground_truch[i];
        //进行前向计算
        Node out = model.forward(input);
        //计算Loss
        Loss l = BCEloss(out,ground_truch);
        //开始反向求导并进行梯度更新
        l.backward(opt);
        //打印损失
        if(i%100==0)
        {
            cout<<"train step: ":<<i<<",loss: "<< l.item()<<endl;
        }
    }
    //保存模型
    model.save("model.weights");
}
````
### 3. 读取预训练模型并进行测试
````c
int main()
{
    ...
    //读取模型文件之前需要先实例化自定义的模型类
    mymodel model = mymodel();
    model.load("model.weights");
    Node input = Node_creater::from_opencv("test.jpg");
    model.forward(input);
    cout<<model.out.argmax()<<endl;
}
````

