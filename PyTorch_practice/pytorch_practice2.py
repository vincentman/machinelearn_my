import torch
import numpy as np
from torch.autograd import Variable

x = torch.rand(5, 3)
print(x.size())
print(x.size()[0])
print(x[0][0])

y = torch.rand(5, 3)
print(x + y)
print(torch.add(x, y))
print(y.add(x))

y.copy_(x)

# -------------backward-------------
x = Variable(torch.ones(2, 2), requires_grad=True)
y = x + 2
print(y.grad_fn)
z = y * y * 3
out = z.mean()
out.backward()
print(x.grad)
print(y.grad)

x = Variable(torch.ones(2, 2), requires_grad=True)
y = 2 * x * x + 2
gradients = torch.FloatTensor([[1, 2], [3, 4]])
# gradients look like weights for backward computing
# y multiplies by gradients and then y.backward
y.backward(gradients, create_graph=True)
print(x.grad)

x = Variable(torch.FloatTensor([[1, 2], [3, 4]]), requires_grad=True)
y = 2 * x * x + 2
y = y.sum()
y.backward()
print(x.grad)

x = Variable(torch.FloatTensor([[1, 2], [3, 4]]), requires_grad=True)
y = 2 * x * x + 2
y.backward(torch.FloatTensor([[1, 2], [3, 4]]))
print(x.grad)

# -------------Variable vs Tensor-------------
x = Variable(torch.ones(4, 5))
y = torch.cos(x)
x_tensor_cos = torch.cos(x.data)
print(y)
print(x_tensor_cos)

print('end')
