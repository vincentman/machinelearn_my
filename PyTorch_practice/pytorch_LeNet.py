import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


# reference
# http://www.cnblogs.com/hellcat/p/6858125.html

class LeNet(nn.Module):
    def __init__(self):
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        # 等价于nn.Model.__init__(self)
        super(LeNet, self).__init__()

        # 输入1通道，输出6通道，卷积核5*5
        # input size = (32, 32), output size = (28, 28)
        self.conv1 = nn.Conv2d(1, 6, 5)
        # 定义卷积层：输入6张特征图，输出16张特征图，卷积核5x5
        # input size = (14, 14), output size = (10, 10)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 定义全连接层：线性连接(y = Wx + b)，16*5*5个节点连接到120个节点上
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # 定义全连接层：线性连接(y = Wx + b)，120个节点连接到84个节点上
        self.fc2 = nn.Linear(120, 84)
        # 定义全连接层：线性连接(y = Wx + b)，84个节点连接到10个节点上
        self.fc3 = nn.Linear(84, 10)

    # 定义向前传播函数，并自动生成向后传播函数(autograd)
    def forward(self, x):
        # 输入x->conv1->relu->2x2窗口的最大池化->更新到x
        # input size = (28, 28), output size = (14, 14), output channel = 6
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 输入x->conv2->relu->2x2窗口的最大池化->更新到x
        # input size = (10, 10), output size = (5, 5), output channel = 16
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # view函数将张量x变形成一维向量形式，总特征数不变，为全连接层做准备
        x = x.view(x.size()[0], -1)
        # 输入x->fc1->relu，更新到x
        # input dim = 16*5*5, output dim = 120
        x = F.relu(self.fc1(x))
        # 输入x->fc2->relu，更新到x
        # input dim = 120, output dim = 84
        x = F.relu(self.fc2(x))
        # 输入x->fc3，更新到x
        # input dim = 84, output dim = 10
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    dict2 = {"key1":1, "key2":2}
    for x in dict2:
        print(x)
    net = LeNet()
    print(net)
    params = list(net.parameters())
    print('len(params): ', len(params))

    # print weight, bias...
    for name, parameters in net.named_parameters():
        print(name, "：", parameters.size())
    input_ = Variable(torch.randn(1, 1, 32, 32))

    # forward propagation
    out = net(input_)
    print('out.size()', out.size())

    # compute loss
    target = Variable(torch.arange(0, 10))
    loss_fn = nn.MSELoss()
    loss = loss_fn(out.squeeze(0), target.float())
    print(loss)

    # add optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    optimizer.zero_grad()  # 效果等同net.zero_grad()
    output = net(input_)
    loss = loss_fn(out.squeeze(0), target.float())
    print("Before backward propagation, parameter's grad: ", net.conv1.bias.grad)

    # backward propagation
    loss.backward()
    print("After backward propagation, parameter's grad: ", net.conv1.bias.grad)
    print("Before backward propagation, parameter's data: ", net.conv1.bias.data)
    optimizer.step()  # update parameter data by gradient
    print("After backward propagation, parameter's data: ", net.conv1.bias.data)
