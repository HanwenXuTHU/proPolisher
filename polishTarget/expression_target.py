import torch
from utils.utils import set_requires_grad


class expression_target():

    def __init__(self, net_path='polishTarget/model/predict_expr_densenet.pth'):
        self.net = torch.load(net_path)
        set_requires_grad(self.net, False)

    def target_loss(self, x):
        output = self.net(x)
        return -output.mean()
