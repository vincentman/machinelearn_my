#
# Modified from PyTorch examples: 
# https://github.com/pytorch/examples/blob/master/mnist/main.py
#

import os
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms  # torchvision contains common utilities for computer vision
from torch.autograd import Variable
import numpy as np

from MNIST_pytorch_vs_tensorflow.pytorch.model import Net


def load_data(train_batch_size, test_batch_size):
    """Fetch MNIST dataset

    MNIST dataset has built-in utilities set up in the `torchvision` package, so we just use the `torchvision.datasets.MNIST` module (http://pytorch.org/docs/master/torchvision/datasets.html#torchvision.datasets.MNIST) to make our lives easier.
    """

    kwargs = {}

    # Fetch training data: total 60000 samples
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../../../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=train_batch_size, shuffle=True, **kwargs)

    # Fetch test data: total 10000 samples
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../../../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=test_batch_size, shuffle=True, **kwargs)

    return (train_loader, test_loader)


def train(model, optimizer, epoch, train_loader, log_interval):
    # State that you are training the model
    model.train()

    # Iterate over batches of data
    for batch_idx, (data, target) in enumerate(train_loader):
        # Wrap the input and target output in the `Variable` wrapper
        data, target = Variable(data), Variable(target)

        # Clear the gradients, since PyTorch accumulates them
        optimizer.zero_grad()

        # Forward propagation
        output = model(data)

        # Calculate negative log likelihood loss
        loss = F.nll_loss(output, target)

        # Backward propagation
        loss.backward()

        # Update the parameters(weight,bias)
        optimizer.step()

        # print log
        if batch_idx % log_interval == 0:
            print('Train set, Epoch {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data.item()))


def test(model, epoch, test_loader):
    # State that you are testing the model; this prevents layers e.g. Dropout to take effect
    model.eval()

    # Init loss & correct prediction accumulators
    test_loss = 0
    correct = 0

    # Optimize the validation process with `torch.no_grad()`
    with torch.no_grad():
        # Iterate over data
        for data, target in test_loader:  # Under `torch.no_grad()`, no need to wrap data & target in `Variable`
            # Forward propagation
            output = model(data)

            # Calculate & accumulate loss
            test_loss += F.nll_loss(output, target, reduction='sum').data.item()

            # Get the index of the max log-probability (the predicted output label)
            # pred = output.data.argmax(1)
            pred = np.argmax(output.data, axis=1)

            # If correct, increment correct prediction accumulator
            # correct += pred.eq(target.data).sum()
            correct = correct + np.equal(pred, target.data).sum()

            # Print log
    test_loss /= len(test_loader.dataset)
    print('\nTest set, Epoch {} , Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(epoch,
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    # Set up training settings from command line options, or use default
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    # Provide seed for the pseudorandom number generator s.t. the same results can be reproduced
    torch.manual_seed(args.seed)

    # Instantiate the model
    model = Net()

    # Choose SGD as the optimizer, initialize it with the parameters & settings
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Load data
    train_loader, test_loader = load_data(args.batch_size, args.test_batch_size)

    # Train & test the model
    for epoch in range(1, args.epochs + 1):
        train(model, optimizer, epoch, train_loader, log_interval=args.log_interval)
        test(model, epoch, test_loader)

    # Save the model for future use
    package_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(package_dir, 'model')
    torch.save(model.state_dict(), model_path)
