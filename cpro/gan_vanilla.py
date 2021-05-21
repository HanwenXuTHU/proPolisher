import torch
import torch.nn as nn
from torch.nn import init
from pro_data import LoadData
from torch.utils.data import DataLoader
import torch.autograd as autograd
from tensor2seq import save_sequence, tensor2seq, reserve_percentage
from tansfer_fasta import csv2fasta
import utils


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028
    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss
    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


def get_infinite_batches(data_loader):
    while True:
        for i, data in enumerate(data_loader):
            yield data


class ResBlock(nn.Module):

    def __init__(self, input_nc, output_nc, kernel_size=5, padding=2, bias=True):
        super(ResBlock, self).__init__()
        model = [nn.ReLU(inplace=False),
                 nn.Conv1d(input_nc, output_nc, kernel_size=kernel_size, padding=padding, bias=bias),
                 nn.ReLU(inplace=False),
                 nn.Conv1d(input_nc, output_nc, kernel_size=kernel_size, padding=padding, bias=bias),
                 ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return x + 0.3*self.model(x)


class Generator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=512, seqL=50, bias=True):
        super(Generator, self).__init__()
        self.ngf, self.seqL = ngf, seqL
        self.first_linear = nn.Linear(seqL*input_nc, seqL*ngf, bias=bias)
        model = [ResBlock(ngf, ngf),
                 ResBlock(ngf, ngf),
                 ResBlock(ngf, ngf),
                 ResBlock(ngf, ngf),
                 ResBlock(ngf, ngf),
                 nn.Conv1d(ngf, output_nc, 1),
                 nn.Softmax(dim=1), ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(self.first_linear(x.view(x.size(0), -1)).reshape([-1, self.ngf, self.seqL]))


class Discriminator(nn.Module):

    def __init__(self, output_nc, ndf=512, seqL=50, bias=True):
        super(Discriminator, self).__init__()
        model = [nn.Conv1d(output_nc, ndf, 1),
                 ResBlock(ndf, ndf),
                 ResBlock(ndf, ndf),
                 ResBlock(ndf, ndf),
                 ResBlock(ndf, ndf),
                 ResBlock(ndf, ndf), ]
        self.model = nn.Sequential(*model)
        self.last_linear = nn.Linear(seqL*ndf, 1, bias=bias)

    def forward(self, x):
        x1 = self.model(x)
        return self.last_linear(x1.view(x1.size(0), -1))


class VanillaLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(VanillaLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss


class WGAN():

    def __init__(self, input_nc, output_nc, seqL=100, lr=1e-4, gpu_ids='0', l1_w=10):
        super(WGAN, self).__init__()
        self.gpu_ids = gpu_ids
        self.l1_w = l1_w
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.generator = Generator(input_nc, output_nc, seqL=seqL)
        self.discriminator = Discriminator(input_nc + output_nc, seqL=seqL)
        self.gan_loss = VanillaLoss()
        if len(gpu_ids) > 0:
            self.generator = self.generator.cuda()
            self.discriminator = self.discriminator.cuda()
            self.gan_loss = self.gan_loss.cuda()
        self.l1_loss = torch.nn.L1Loss()
        self.optim_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.9))
        self.optim_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))

    def backward_g(self, fake_x):
        for p in self.discriminator.parameters():
            p.requires_grad = False
        self.generator.zero_grad()
        self.fake_inputs = self.generator(fake_x)
        pred_fake = self.discriminator(torch.cat((fake_x, self.fake_inputs), 1))
        self.g_loss = self.gan_loss(pred_fake, True)
        self.g_l1 = self.l1_w*self.l1_loss(fake_x, self.fake_inputs)
        self.g_total_loss = self.g_loss + self.g_l1
        self.g_total_loss.backward()
        self.optim_g.step()

    def backward_d(self, real_x, fake_x):
        for p in self.discriminator.parameters():
            p.requires_grad = True
        self.fake_inputs = self.generator(fake_x)
        fakeAB = torch.cat((fake_x, self.fake_inputs), 1)
        realAB = torch.cat((fake_x, real_x), 1)
        self.discriminator.zero_grad()
        pred_fake, pred_real = self.discriminator(fakeAB), self.discriminator(realAB)
        self.d_loss = self.gan_loss(pred_fake, False) + self.gan_loss(pred_real, True)
        self.d_total_loss = self.d_loss
        self.d_total_loss.backward()
        self.optim_d.step()


def main():
    train_data, test_data = DataLoader(LoadData(is_train=True, path='data/ecoli_test_ability_reserve-35.csv', split_r=1),
                                       batch_size=32, shuffle=True), DataLoader(LoadData(is_train=True, path='data/ecoli_test_ability_reserve-35.csv', split_r=1), batch_size=32)
    train_data = get_infinite_batches(train_data)
    logger = utils.get_logger()
    n_critics = 1
    n_iters = 10000
    model = WGAN(input_nc=4, output_nc=4, seqL=100, l1_w=30)
    d_loss, g_loss, g_l1 = 0, 0, 0
    tensorSeq, tensorInput, tensorRealB = [], [], []
    for j, eval_data in enumerate(test_data):
        with torch.no_grad():
            tensorSeq.append(model.generator(eval_data['in']))
            tensorInput.append(eval_data['in'])
            tensorRealB.append(eval_data['out'])
    logger.info('Testing: reserve percentage: {}%'.format(reserve_percentage(tensorInput, tensorSeq)))
    csv_name = save_sequence(tensorSeq, tensorInput, tensorRealB, 'wgan_')
    csv2fasta(csv_name, 'cache/gen_', 'iter_{}'.format(0))
    logger.info('Saving Test Succeed!')
    torch.save(model.generator, 'check_points/' + 'net_G_' + str(0) + '.pth')
    torch.save(model.discriminator, 'check_points/' + 'net_D_' + str(0) + '.pth')
    logger.info('Saving Model Test Succeed!')
    for i in range(n_iters):
        #train discriminators
        for j in range(n_critics):
            _data = train_data.__next__()
            model.backward_d(_data['out'], _data['in'])
            d_loss += float(model.d_loss)
        model.backward_g(_data['in'])
        g_loss += float(model.g_loss)
        g_l1 += float(model.g_l1)
        if i % 100 == 99:
            tensorSeq, tensorInput, tensorRealB = [], [], []
            for j, eval_data in enumerate(test_data):
                with torch.no_grad():
                    tensorSeq.append(model.generator(eval_data['in']))
                    tensorInput.append(eval_data['in'])
                    tensorRealB.append(eval_data['out'])
            logger.info('Training: iters: {}, dloss: {}, gloss:{}, gl1:{}'.format(i, d_loss / 100 / n_critics, g_loss / 100, g_l1 / 100))
            logger.info('Testing: reserve percentage: {}%'.format(reserve_percentage(tensorInput, tensorSeq)))
            d_loss, g_loss, g_l1 = 0, 0, 0
            csv_name = save_sequence(tensorSeq, tensorInput, tensorRealB, 'wgan_')
            relat_v = utils.relation_reserve(csv_name, st=[64], ed=[70])
            logger.info('relation correct ratio: {}'.format(relat_v))
            A_dict_valid, A_dict_ref, T_dict_valid, T_dict_ref = utils.polyAT_freq(csv_name, 'data/ecoli_100_space_fix.csv')
            logger.info('polyA valid AAAAA:{} AAAAAA:{} AAAAAAA:{} AAAAAAAA:{}'.format(A_dict_valid['AAAAA'],
                                                                                 A_dict_valid['AAAAAA'],
                                                                                 A_dict_valid['AAAAAAA'],
                                                                                 A_dict_valid['AAAAAAAA']))
            logger.info('polyA ref AAAAA:{} AAAAAA:{} AAAAAAA:{} AAAAAAAA:{}'.format(A_dict_ref['AAAAA'],
                                                                                 A_dict_ref['AAAAAA'],
                                                                                 A_dict_ref['AAAAAAA'],
                                                                                 A_dict_ref['AAAAAAAA']))
            logger.info('polyT valid TTTTT:{} TTTTTT:{} TTTTTTT:{} TTTTTTTT:{}'.format(T_dict_valid['TTTTT'],
                                                                                 T_dict_valid['TTTTTT'],
                                                                                 T_dict_valid['TTTTTTT'],
                                                                                 T_dict_valid['TTTTTTTT']))
            logger.info('polyT ref TTTTT:{} TTTTTT:{} TTTTTTT:{} TTTTTTTT:{}'.format(T_dict_ref['TTTTT'],
                                                                               T_dict_ref['TTTTTT'],
                                                                               T_dict_ref['TTTTTTT'],
                                                                               T_dict_ref['TTTTTTTT']))
            csv2fasta(csv_name, 'cache/gen_', 'iter_{}'.format(i))
            torch.save(model.generator, 'check_points/' + 'net_G_' + str(i) + '.pth')
            torch.save(model.discriminator, 'check_points/' + 'net_D_' + str(i) + '.pth')


if __name__ == '__main__':
    main()
