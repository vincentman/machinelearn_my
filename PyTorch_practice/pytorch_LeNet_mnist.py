import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from PyTorch_practice.pytorch_LeNet import LeNet
import numpy as np
from PIL import Image


def load_data(train_batch_size, test_batch_size):
    # Fetch training data: total 60000 samples
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../../../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Resize((32, 32)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=train_batch_size, shuffle=True)

    # Fetch test data: total 10000 samples
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../../../data', train=False, transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=test_batch_size, shuffle=True)

    return (train_loader, test_loader)


def train(model, optimizer, epoch, train_loader, log_interval):
    # State that you are training the model
    model.train()

    # define loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Iterate over batches of data
    for batch_idx, (data, target) in enumerate(train_loader):
        # Wrap the input and target output in the `Variable` wrapper
        data, target = Variable(data), Variable(target)

        # Clear the gradients, since PyTorch accumulates them
        optimizer.zero_grad()

        # Forward propagation
        output = model(data)

        loss = loss_fn(output, target)

        # Backward propagation
        loss.backward()

        # Update the parameters(weight,bias)
        optimizer.step()

        # print log
        if batch_idx % log_interval == 0:
            print('Train set, Epoch {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                loss.data.item()))
                # loss.data[0]))


def test(model, epoch, test_loader):
    # State that you are testing the model; this prevents layers e.g. Dropout to take effect
    model.eval()

    # Init loss & correct prediction accumulators
    test_loss = 0
    correct = 0

    # define loss function
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    # Optimize the validation process with `torch.no_grad()`
    with torch.no_grad():
        # Iterate over data
        for data, target in test_loader:  # Under `torch.no_grad()`, no need to wrap data & target in `Variable`
            # Forward propagation
            output = model(data)

            # Calculate & accumulate loss
            test_loss += loss_fn(output, target).data.item()

            # Get the index of the max log-probability (the predicted output label)
            pred = output.data.argmax(1)
            # pred = np.argmax(output.data, axis=1)

            # If correct, increment correct prediction accumulator
            correct += pred.eq(target.data).sum()
            # correct = correct + np.equal(pred, target.data).sum()

    # Print log
    test_loss /= len(test_loader.dataset)
    print('\nTest set, Epoch {} , Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(epoch,
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# Provide seed for the pseudorandom number generator s.t. the same results can be reproduced
torch.manual_seed(123)

model = LeNet()

lr = 0.01
momentum=0.5
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

train_batch_size = 64
test_batch_size = 1000
train_loader, test_loader = load_data(train_batch_size, test_batch_size)

epochs = 10
log_interval = 100
for epoch in range(1, epochs + 1):
    train(model, optimizer, epoch, train_loader, log_interval=log_interval)
    test(model, epoch, test_loader)

# Load & transform image
ori_img = Image.open('test_4.png').convert('L')
t = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
img = torch.autograd.Variable(t(ori_img).unsqueeze(0))
ori_img.close()

# Predict
model.eval()
output = model(img)
pred = output.data.max(1, keepdim=True)[1][0][0]
print('Prediction: {}'.format(pred))