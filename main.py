# 这是一个示例 Python 脚本。
# 此脚本使用Bert对Conll2003数据集进行命名实体识别
# https://blog.csdn.net/qq_44827933/article/details/134518079

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertForPreTraining
import torch.nn as nn
import time
from tqdm import tqdm
from matplotlib import pyplot as plt
from seqeval.metrics import accuracy_score,f1_score


def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。


def readFile(name):
    """
    读取数据
    返回:
    [
    [token,token,token,token],
    [token,token,token,token],
    ]
    [
    [label,label,label,label],
    [label,label,label,label],
    ]
    """
    data = []
    label = []
    dataSentence = []
    labelSentence = []
    with open(name, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if not line.strip():
                data.append(dataSentence)
                label.append(labelSentence)
                dataSentence = []
                labelSentence = []
            else:
                content = line.strip().split()
                dataSentence.append(content[0].lower())
                labelSentence.append(content[-1])
        return data, label


def label2index(label):
    """
    # 读取标签数据，构建标签字典

    # 返回： 标签个数，标签列表
    """
    label2index = {}
    for sentence in label:
        for i in sentence:
            if i not in label2index:
                label2index[i] = len(label2index)
    return label2index,list(label2index)


class Dataset(Dataset):
    def __init__(self, data, label, labelIndex, tokenizer, maxlength):
        self.data = data
        self.label = label
        self.labelIndex = labelIndex
        self.tokernizer = tokenizer
        self.maxlength = maxlength

    def __getitem__(self, item):
        thisdata = self.data[item]
        thislabel = self.label[item][:self.maxlength]
        thisdataIndex = self.tokernizer.encode(thisdata, add_special_tokens=True, max_length=self.maxlength + 2,
                                               padding="max_length", truncation=True, return_tensors="pt")
        thislabelIndex = [self.labelIndex['O']] + [self.labelIndex[i] for i in thislabel] + [self.labelIndex['O']] * (
                    self.maxlength + 1 - len(thislabel))
        thislabelIndex = torch.tensor(thislabelIndex)
        return thisdataIndex[-1], thislabelIndex,len(thislabel)

    def __len__(self):
        return self.data.__len__()


# 建模
class BertModel(nn.Module):
    """
    这个类继承自 PyTorch 的 nn.Module，这使得它可以作为神经网络的一部分使用。
    """
    def __init__(self, classnum, criterion):
        """
        :param classnum: 分类的数量，即模型输出的类别数量
        :param criterion: 损失函数，用于计算模型输出与真实标签之间的误差
        """
        super().__init__()
        self.bert = BertForPreTraining.from_pretrained('bert-base-uncased').bert # 加载预训练的 BERT 模型
        self.classifier = nn.Linear(768, classnum)# 创建线性分类层，将 BERT 的输出（768 维）映射到类别数量 classnum。
        self.criterion = criterion # 损失函数

    def forward(self, batchdata, batchlabel=None):
        """
        前向传播方法
        :param batchdata:分词处理后的张量
        :param batchlabel:可选参数:真实标签
        :return:
        """
        bertOut=self.bert(batchdata) # 将输入数据传入 BERT 模型，获得输出
        # bertOut[0]: 包含每个 token 的隐藏状态，形状为 [batchsize, maxlength, 768]。
        # bertOut[1]: 通常是输入序列的整体表示，形状为 [batchsize, 768]
        bertOut0,bertOut1=bertOut[0],bertOut[1]#字符级别bertOut[0].size()=torch.Size([batchsize, maxlength, 768]),篇章级别bertOut[1].size()=torch.Size([batchsize,768])
        pre=self.classifier(bertOut0) # 将字符级别的输出（每个 token 的表示）传入分类层，得到预测结果

        if batchlabel is not None:
            # 有真实标签
            loss=self.criterion(pre.reshape(-1,pre.shape[-1]),batchlabel.reshape(-1))
            return loss
        else:
            return torch.argmax(pre,dim=-1)
        # print(self.bert(batch_index))


# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':

    # 超参数
    batchsize = 64
    epoch = 10
    maxlength = 75
    lr = 0.01
    weight_decay = 0.00001

    # 读取数据
    trainData, trainLabel = readFile('./data/train.txt')
    devData, devLabel = readFile('./data/dev.txt')
    testData, testLabel = readFile('./data/test.txt')

    # 构建词表
    labelIndex,indexLabel = label2index(trainLabel)
    # print(labelIndex)

    # 构建数据集,迭代器
    if hasattr(torch.cuda, 'empty_cache'): # 检查torch.cuda是否有empty_cache，有则清除缓存
        torch.cuda.empty_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu" # 检查cuda是否可用
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') # 加载 BERT 模型的分词器 bert-base-uncased
    trainDataset = Dataset(trainData, trainLabel, labelIndex, tokenizer, maxlength) # 创建训练数据集
    trainDataloader = DataLoader(trainDataset, batch_size=batchsize, shuffle=False) # 创建训练数据加载器 batch_size 每个批次的样本数量 shuffle=False 不打乱数据顺序
    devDataset = Dataset(devData, devLabel, labelIndex, tokenizer, maxlength) # 创建验证数据集
    devDataloader = DataLoader(devDataset, batch_size=batchsize, shuffle=False) # 创建验证数据集加载器

    # 建模
    criterion = nn.CrossEntropyLoss()
    model = BertModel(len(labelIndex), criterion).to(device)
    # print(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    # #绘图准备
    epochPlt = []
    trainLossPlt = []
    devAccPlt = []
    devF1Plt = []

    # 训练验证
    for e in range(epoch):
        # 训练
        time.sleep(0.1)
        print(f'epoch:{e + 1}')
        epochPlt.append(e + 1)
        epochloss = 0
        model.train()
        for batchdata, batchlabel, batchlen in tqdm(trainDataloader, total=len(trainDataloader), leave=False,
                                                    desc="train"):
            batchdata = batchdata.to(device)
            batchlabel = batchlabel.to(device)
            loss = model.forward(batchdata, batchlabel)
            epochloss += loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        epochloss /= len(trainDataloader)
        trainLossPlt.append(float(epochloss))
        print(f'loss:{epochloss:.5f}')
        # print(batchdata.shape)
        # print(batchlabel.shape)

        # 验证
        time.sleep(0.1)
        epochbatchlabel = []
        epochpre = []
        model.eval()
        for batchdata, batchlabel, batchlen in tqdm(devDataloader, total=len(devDataloader), leave=False, desc="dev"):
            batchdata = batchdata.to(device)
            batchlabel = batchlabel.to(device)
            pre = model.forward(batchdata)
            pre = pre.cpu().numpy().tolist()
            batchlabel = batchlabel.cpu().numpy().tolist()

            for b, p, l in zip(batchlabel, pre, batchlen):
                b = b[1:l + 1]
                p = p[1:l + 1]
                b = [indexLabel[i] for i in b]
                p = [indexLabel[i] for i in p]
                epochbatchlabel.append(b)
                epochpre.append(p)
            # print(pre)
        acc = accuracy_score(epochbatchlabel, epochpre)
        f1 = f1_score(epochbatchlabel, epochpre)
        devAccPlt.append(acc)
        devF1Plt.append(f1)
        print(f'acc:{acc:.4f}')
        print(f'f1:{f1:.4f}')

        # 绘图
        # print(epochPlt, trainLossPlt,devAccPlt,devF1Plt)
        plt.plot(epochPlt, trainLossPlt)
        plt.plot(epochPlt, devAccPlt)
        plt.plot(epochPlt, devF1Plt)
        plt.ylabel('loss/Accuracy/f1')
        plt.xlabel('epoch')
        plt.legend(['trainLoss', 'devAcc', 'devF1'], loc='best')
        plt.show()

    print_hi('PyCharm')