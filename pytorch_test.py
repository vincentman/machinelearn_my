import torch
import numpy as np

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
cat = torch.stack((randn, randn), 1)

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

# switch between numpy array and torch tensor
x = np.array([1, 2, 3])
x = torch.from_numpy(x)
x[0] = -1
x = x.numpy()

cuda_isavail = torch.cuda.is_available()


print('The end')
