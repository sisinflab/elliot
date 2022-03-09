import torch


class MyLinear(torch.nn.Module):
    def __init__(self, weights, bias):
        super(MyLinear, self).__init__()

        self.linear = torch.nn.Linear(weights.shape[0], weights.shape[1])
        with torch.no_grad():
            self.linear.weight.copy_(weights)
            self.linear.bias.copy_(torch.squeeze(bias, dim=1))

    def forward(self, x):
        x = self.linear(x)
        return x
