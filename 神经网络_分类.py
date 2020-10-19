import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

n_data = torch.ones(100,2)
# x0包括横坐标和纵坐标
x0 = torch.normal(2*n_data, 1)  # class0 x data (tensor), shape = (100,2)
# y0则是作为点的值,这里y0是0,y1是1
y0 = torch.zeros(100)       # class0 y data (tensor), shape = (100, 1)
x1 = torch.normal(-2*n_data, 1)  # class1 x data (tensor), shape = (100, 1)
y1 = torch.ones(100)            # class1 y data (tensor), shape  = (100, 1)
# x作为输入, 接收从隐藏层过来的n_hidden个输入信息,输出信息即是1,这里输出到n_output
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # shape (200, 2) FloatTensor = 32-bit floating
# y作为标签, 并且pytorch默认要求标签的类型为LongTensor
y = torch.cat((y0, y1), ).type(torch.LongTensor)    # shape (200,) LongTensor = 64-bit integer


# torch can only train on Variable, so convert them to Variable
x, y = Variable(x), Variable(y)

# # 画散点图
# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
# plt.show()

# method 1
# 定义自己的网络
class Net(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_output):
        super(Net, self).__init__() # 这一步为必须步骤

        # 隐藏层
        self.hidden = torch.nn.Linear(n_features, n_hidden) # 这就是一层网络
        # 预测的神经层
        self.predict = torch.nn.Linear(n_hidden, n_output) 

    # 定义前向传递的过程
    def forward(self, x):       # 这里是搭建网络的过程

        # 这里使用hidden layer加工一下输入信息,即self.hidden(x)
        x = F.relu(self.hidden(x))        # 激励函数激活一下信息    输入
        x = self.predict(x)     # 输出  (此处不用激励函数是因为在大多数回归问题中, 值的分布是负无穷到正无穷,再使用分段函数的话会截断一部分值)
        return x

net = Net(2, 10, 2) # 第一个，第三个参数为输入，输出，均为两个参数。二分类为[1,0]表示电脑认为应该分为第一类,同样 [0,1]表示计算机认为应该属于第二类
# print(net)

# 快速搭建
# method 2
net2 = torch.nn.Squential(
            torch.nn.Linear(2,10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 2)
        )
# 这个神经网络与上面的method1是一样的

print(net)
print(net2)


plt.ion()       # something about plotting
plt.show()

optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()     # 这个函数是用于多分类问题

for t in range(100):
    out = net(x)

    loss = loss_func(out, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if(t % 2 == 0):
        plt.cla()
        prediction = torch.max(F.softmax(out), 1)[1]
        prey_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:,0], x.data.numpy()[:,1], c=prey_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(prey_y == target_y) / 200
       # plt.text(1.5, -4, 'Accuracy=.2f%' % accuracy, fontdict={'size':20, 'color':'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
















