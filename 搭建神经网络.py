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
        #   接收从隐藏层过来的n_hidden个输入信息,输出信息即是1,这里输出到n_output
        self.predict = torch.nn.Linear(n_hidden, n_output) 

    # 定义前向传递的过程
    def forward(self, x):       # 这里是搭建网络的过程
        # 激励函数就是一系列非线性的函数
        # 常用的激励函数包括 ：relu, sigmoid, tanh, softplus
        # 需要激励函数的原因： 神经网络每一层的结果都是线性的， 有些复杂的问题无法用线性
        # 的模型去学习描述，所以需要非线性的激励函数去处理。

        # 这里使用hidden layer加工一下输入信息,即self.hidden(x)
        # x = F.relu(self.hidden(x))        # 激励函数激活一下信息    输入
        x = F.tanh(self.hidden(x))      # 使用tanh + lr=0.3貌似可以产生更接近的结果
        x = self.predict(x)     # 输出  (此处不用激励函数是因为在大多数回归问题中, 值的分布是负无穷到正无穷,再使用分段函数的话会截断一部分值)
        return x


net = Net(1, 10, 1)
# print(net)      # 可以打印看看网络搭建是什么样子
# 输出结果:Net(
# (hidden): Linear(in_features=1, out_features=10, bias=True)
# (predict): Linear(in_features=10, out_features=1, bias=True)
# )

# 可视化 
# 首先设置matplotlib变为实时打印的过程
plt.ion()
plt.show()

# 搭建完了神经网络,下面就是如何优化
# 使用SGD来优化, 需要传入神经网络的全部参数,即net.parameters()
# lr为学习效率      一般小于1
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
loss_func = torch.nn.MSELoss()      # 这里MSELoss为均方差, 使用均方差来处理回归问题就足够了.分类问题使用另外的loss_func

# 开始训练
for t in range(1000):    # 训练100次
    prediction = net(x)

    # 计算误差, 即prediction(预测值)和y(真实值)之间的误差
    # 且必须是prediction在前, 真实值在后,不然有时候会出错
    loss = loss_func(prediction, y)
    
    # 下面三步就是优化
    optimizer.zero_grad()       #先将参数梯度归零,因为每次计算loss之后梯度都会保留在optimizer里面,所以要先清零
    loss.backward()     # 然后进行反向传递
    optimizer.step()    

    # print(loss.data[0])  #invalid index of a 0-dim tensor. Use tensor.item() to convert a 0-dim tensor to a Python number
    # print(type(loss.item()))        # loss.item是一个函数, loss.item()是函数的返回值
    # 结果: <class 'float'>

    # 可视化
    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        # 由于是实时打印,每一次都要重新打印所有的点
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        # plt.text(0.5, 0, 'loss=%.4f' % loss.data[0], fontdict={'size': 20, 'color':'red'})
        # invalid index of a 0-dim tensor. Use tensor.item() to convert a 0-dim tensor to a Python number
        plt.text(0.5, 0, 'loss=%.4f' % loss.item(), fontdict={'size': 20, 'color':'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()

