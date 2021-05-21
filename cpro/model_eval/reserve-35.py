import torch
import torch.nn as nn
from torch.nn import init
from pro_data import LoadData
from torch.utils.data import DataLoader
import torch.autograd as autograd
from tensor2seq import save_sequence, tensor2seq, reserve_percentage
from tansfer_fasta import csv2fasta
from wgan import Generator, ResBlock


def main():
    eval_data = DataLoader(LoadData(path='data/test-input-35.csv', is_train=False), batch_size=32)
    load_iter = 2199
    model_g = torch.load('../check_points/G_{}.pth'.format(load_iter))
    tensorSeq, tensorInput, tensorRealB = [], [], []
    for j, eval_data in enumerate(eval_data):
        with torch.no_grad():
            tensorSeq.append(model_g(eval_data['in']))
            tensorInput.append(eval_data['in'])
            tensorRealB.append(eval_data['out'])
    print('Testing: reserve percentage: {}%'.format(reserve_percentage(tensorInput, tensorSeq)))
    csv_name = save_sequence(tensorSeq, tensorInput, tensorRealB, 'wgan_')
    csv2fasta(csv_name, 'results/gen_-35_', 'iter_{}'.format(load_iter))

if __name__ == '__main__':
    main()