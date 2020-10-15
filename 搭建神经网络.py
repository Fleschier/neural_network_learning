import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

# unsqueeze函数可以将linspace生成的一位数据转化为二维数据
# input (Tensor) – the input tensor.
# dim (int) – the index at which to insert the singleton dimension

# 例子：
# >>> x = torch.tensor([1, 2, 3, 4])
# >>> torch.unsqueeze(x, 0)
# tensor([[ 1,  2,  3,  4]])
# >>> torch.unsqueeze(x, 1)
# tensor([[ 1],
#         [ 2],
#         [ 3],
#         [ 4]])
x = torch.unsqueeze(torch.linspace(-1,1,100), dim=1)
y = x.pow(2) + 0.2*torch.rand(x.size())

x,y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()

# 定义自己的网络
class Net(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_output):
        super(Net, self).__init__() # 这一步为必须步骤
        # n_features就是这一层的输入
        # n_hidden多少个隐藏层的神经元
        # n_output 多少个输出

        # 隐藏层
        self.hidden = torch.nn.Linear(n_features, n_hidden) # 这就是一层网络
        # 预测的神经层
        self.predict = torch.nn.Linear(n_hidden, n_output) 

    # 定义前向传递的过程
    def forward(self, x):       # 这里是搭建网络的过程
        # 激励函数就是一系列非线性的函数
        # 常用的激励函数包括 ：relu, sigmoid, tanh, softplus
        # 需要激励函数的原因： 神经网络每一层的结果都是线性的， 有些复杂的问题无法用线性
        # 的模型去学习描述，所以需要非线性的激励函数去处理。
        x = F.relu(self.hidden(x))        # 激励函数激活一下信息
