import torch
import numpy as np
from torch.autograd import Variable
from torch.optim import SGD

# reference
# https://fgc.stpi.narl.org.tw/activity/videoDetail/4b1141305d9cd231015d9d0992ef0030

ones = torch.ones(5, 3)
zeros = torch.zeros(5, 3)
eye = torch.eye(4)
rand = torch.rand(5, 3)
randn = torch.randn(5, 3)

# shape: 10x3
cat = torch.cat((randn, randn), 0)
# shape: 5x6
cat = torch.cat((randn, randn), 1)

# shape: 5x2x3
stack = torch.stack((randn, randn), 1)

# shape: 5x1x3
randn2 = torch.randn(5, 1, 3)
# shape: 5x3
squeeze = randn2.squeeze(1)
# shape: 5x1x3
unsqueeze = squeeze.unsqueeze(1)

# shape: 5x1x3
add = randn2 + unsqueeze

# shape: 4x4
rand2 = torch.rand(4, 4)
# shape: 2x2x4
view = rand2.view(2, 2, -1)

# shape: 3x1
x = torch.Tensor([[1], [2], [3]])
# shape: 3x4
expand = x.expand(3, 4)

# compute on GPU
# expand.cuda()
# compute on CPU
expand.cpu()
print('cuda.is_available(): ', torch.cuda.is_available())

# switch between numpy array and torch tensor: share memory
x = np.array([1, 2, 3])
x = torch.from_numpy(x)
x[0] = -1
x = x.numpy()

m1 = torch.ones(5, 3)
m2 = torch.ones(5, 3)


# demo optimizer and backward propagation
a = Variable(m1, requires_grad=True)
b = Variable(m2, requires_grad=True)
optimizer = SGD([a, b], lr=0.1)
for _ in range(10):        # 我們示範更新10次
    loss = (a + b).sum()   # 假設a + b就是我們的loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()       # 更新

print('The end')
