import torch
import torch.nn as nn


class ResBlock(nn.Module):

    def __init__(self, input_nc, output_nc, kernel_size=5, padding=2, bias=True):
        super(ResBlock, self).__init__()
        model = [nn.ReLU(inplace=False),
                 nn.Conv1d(input_nc, output_nc, kernel_size=kernel_size, padding=padding, bias=bias),
                 nn.ReLU(inplace=False),
                 nn.Conv1d(output_nc, output_nc, kernel_size=kernel_size, padding=padding, bias=bias),
                 ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return x + 0.3*self.model(x)


class probLayer(nn.Module):

    def __init__(self, input_nc=4, ndf=512, kernel_size=7, paddinng=3):
        super(probLayer, self).__init__()
        model = [nn.Conv1d(in_channels=input_nc, out_channels=ndf, kernel_size=kernel_size, padding=paddinng),
                 ResBlock(input_nc=ndf, output_nc=ndf, kernel_size=kernel_size, padding=paddinng),
                 ResBlock(input_nc=ndf, output_nc=ndf, kernel_size=kernel_size, padding=paddinng),
                 nn.Conv1d(in_channels=ndf, out_channels=1, kernel_size=kernel_size, padding=paddinng)]
        self.model = nn.Sequential(*model)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = torch.squeeze(self.model(x))
        return self.sigmoid(x1)