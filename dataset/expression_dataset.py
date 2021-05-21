import xlrd
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import ConcatDataset
import torch
import random


class Dataset(object):

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])


class SeqDataset(Dataset):

    def __init__(self, path='dataset/data/ecoli_expr_rna.xlsx', isTrain=True, isGpu=True, isZero=True, split_r=1):
        self.path = path
        book1 = xlrd.open_workbook(self.path)
        sheet = book1.sheet_by_name('ecoli_expr_rna')
        nrows = sheet.nrows
        random.seed(0)
        index = list(np.arange(nrows))
        index = index[1 : ]
        random.shuffle(index)
        self.pSeq = []
        self.expr = []
        self.isReal = []
        self.backbone_seq = []
        self.isTrain = isTrain
        self.split_r = split_r
        self.isGpu = isGpu
        w_expr = []
        for i in range(1, nrows - 1):
            w_expr.append(sheet.cell(index[i], 2).value)
        w_expr = np.asarray(w_expr)
        maxE = 1
        minE = 0
        if self.isTrain:
            start, end = 1, int(nrows*self.split_r) - 1
        else:
            start, end = int(nrows*self.split_r) - 1, nrows - 1
        for i in range(start, end):
            if isZero == True:
                self.pSeq.append(self.oneHot(sheet.cell(index[i], 0).value))
                self.backbone_seq.append(self.backbone(sheet.cell(index[i], 0).value))
                self.isReal.append(sheet.cell(index[i], 1).value)
                self.expr.append((sheet.cell(index[i], 2).value - minE)/(maxE - minE))
            else:
                if sheet.cell(index[i], 2).value > 0:
                    self.pSeq.append(self.oneHot(sheet.cell(index[i], 0).value))
                    self.backbone_seq.append(self.backbone(sheet.cell(index[i], 0).value))
                    self.isReal.append(sheet.cell(index[i], 1).value)
                    self.expr.append((sheet.cell(index[i], 2).value - minE)/(maxE - minE))

    def oneHot(self, sequence):
        oh_dict = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
        oh = np.zeros([4, len(sequence)])
        for i in range(len(sequence)):
            oh[oh_dict[sequence[i]], i] = 1
        return oh

    def backbone(self, sequence):
        oh = np.zeros([4, len(sequence)])
        for i in range(len(sequence)):
            oh[:, i] = np.random.rand(4)
        return oh

    def __getitem__(self, item):
        X = self.pSeq[item][:, :]
        Y = self.backbone_seq[item]
        X = transforms.ToTensor()(X)
        X = torch.squeeze(X)
        X = X.float()
        Y = transforms.ToTensor()(Y)
        Y = torch.squeeze(Y)
        Y = Y.float()
        if self.isGpu:
            X, Y= X.cuda(), Y.cuda()
        return {'x': X,'backbone': Y}

    def __len__(self):
        return len(self.isReal)