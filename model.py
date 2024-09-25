import torch
import torch.nn.functional as F
from torch.nn import Conv2d, Dropout2d, Linear


class Net(torch.nn.Module):
    def __init__(self):
        """
        Initialize the Net model with two convolutional layers,
        a dropout layer, and two fully connected linear layers.
        """

        super(Net, self).__init__()
        self.conv1 = Conv2d(1, 10, kernel_size=5)
        self.conv2 = Conv2d(10, 20, kernel_size=5)
        self.dropout = Dropout2d(0.2)
        self.linear1 = Linear(320, 20)
        self.linear2 = Linear(20, 10)

    def main(self, x):
        """
        Define forward propagation of the network.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor containing log-probabilities for each class.
        """

        x = F.max_pool2d(self.conv1(x), 2)
        x = F.relu(x)
        x = self.dropout(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = x.view(-1, 320)
        x = F.relu(self.linear1(x))
        x = F.dropout(x, training=self.training)
        x = self.linear2(x)
        return F.log_softmax(x, dim=1)

    def forward(self, x):
        """
        Forward pass that calls main function.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Log-probabilities of each class for the input tensor.
        """

        return self.main(x)
