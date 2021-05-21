import torch
import torch.nn as nn
from probLayer.cnn_model import probLayer
from utils.utils import set_requires_grad, get_logger
from cpro.wgan import *
from dataset.expression_dataset import SeqDataset
from polishTarget import SeqRegressionModel


class proPolisher():

    def __init__(self,
                 input_nc=4,
                 generator_path='cpro/check_points/net_G_3999.pth',
                 prob_ndf=512,
                 prob_kernel=7,
                 prob_padding=3,
                 sample_num=50,
                 target='expression',
                 target_path='polishTarget/model/predict_expr_densenet.pth',
                 save_path = 'probLayer/model/',
                 lr=0.0001,
                 gpu_ids='0',
                 batch_size=10,
                 epochs=100):
        self.input_nc = input_nc
        self.prob_layer = probLayer(input_nc=input_nc, ndf=prob_ndf, kernel_size=prob_kernel, paddinng=prob_padding)
        self.generator = torch.load(generator_path)
        self.sample_num = sample_num
        self.epochs = epochs
        self.save_path = save_path
        if len(gpu_ids) > 0:
            self.gpu = True
            self.prob_layer = self.prob_layer.cuda()
        self.batch_size = batch_size
        set_requires_grad(self.generator, False)
        if target == 'expression':
            from polishTarget.expression_target import expression_target
            self.target = expression_target(net_path=target_path)
        self.optimizer = torch.optim.Adam(params=self.prob_layer.parameters(), lr=lr)
        self.logger = get_logger()

    def get_results(self, x ,backbone):
        sample_list = []
        self.probs = self.prob_layer(x)
        for i in range(self.sample_num):
            self.masks = torch.bernoulli(self.probs)
            self.masks_expand = self.masks.unsqueeze(dim=1)
            new_shape = [self.masks.size(0), self.input_nc, self.masks.size(1)]
            self.masks_expand = self.masks_expand.expand(new_shape)
            x_mask = torch.multiply(x, self.masks_expand)
            mask_inverse = torch.unsqueeze(torch.tensor(1.0).cuda().expand_as(self.masks) - self.masks, 1)
            new_shape = [mask_inverse.size(0), self.input_nc, mask_inverse.size(2)]
            backbone_mask = torch.multiply(mask_inverse.expand(*new_shape), backbone)
            sample_list.append(self.generator(x_mask + backbone_mask))
        samples = torch.cat(sample_list, dim=0)
        return samples

    def backward_prob(self, x, backbone):
        samples = self.get_results(x, backbone)
        self.prob_layer.zero_grad()
        self.loss = self.target.target_loss(samples)
        self.loss.backward()
        self.optimizer.step()

    def training(self):
        self.dataset_train = DataLoader(dataset=SeqDataset(isTrain=True, isGpu=self.gpu), batch_size=self.batch_size,
                                   shuffle=True)
        self.dataset_test = DataLoader(dataset=SeqDataset(isTrain=True, isGpu=self.gpu), batch_size=self.batch_size,
                                       shuffle=False)
        for i in range(self.epochs):
            train_loss, num = 0, 0
            debug = 0
            for train_loader in self.dataset_train:
                input_x, input_backbone = train_loader['x'], train_loader['backbone']
                self.backward_prob(input_x, input_backbone)
                train_loss += float(self.loss)
                num += 1
            train_loss = train_loss/num
            test_loss, num = 0, 0
            for test_loader in self.dataset_test:
                input_x, input_backbone = test_loader['x'], test_loader['backbone']
                with torch.no_grad():
                    test_samples = self.get_results(input_x, input_backbone)
                    test_loss = self.target.target_loss(test_samples)
                test_loss += float(test_loss)
                num += 1
            test_loss = test_loss / num
            self.logger.info('iter:{} train loss:{} test loss:{}'.format(i, train_loss, test_loss))
            torch.save(self.prob_layer, self.save_path + 'prob_layer_{}.pth'.format(i))


def main():
    polisher = proPolisher()
    polisher.training()


if __name__ == '__main__':
    main()








