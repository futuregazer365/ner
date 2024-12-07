## 数据集简介

CoNLL-2003

**dev.txt(validation.txt)**

验证数据集。用于在训练过程中评估模型的性能，以便调整超参数和防止过拟合。通常在每个训练周期（epoch）后评估模型。

**train.txt**

训练数据集。用于训练模型，通过调整模型参数以最小化损失函数，从而学习输入数据与输出标签之间的关系。

**test.txt**

测试数据集。用于在模型训练和验证完成后评估模型的最终性能。测试集应独立于训练和验证集，以确保评估结果的可靠性。

## **数据格式**

BIO标注法，B-begin，代表实体的开头；I-inside，代表实体的中间或结尾；O-outside，代表不属于实体。
实体分为四类：人名（PER）、地名（LOC）、组织名（ORG）、其他实体名（MISC）。
共组成九种实体标签：

| 词             | 词性 | 语法块 | 实体标签 |
| -------------- | ---- | ------ | -------- |
| SOCCER         | NN   | B-NP   | O        |
| JAPAN          | NNP  | B-NP   | B-LOC    |
| GET            | VB   | B-VP   | O        |
| CRICKET        | NNP  | B-NP   | O        |
| LEICESTERSHIRE | NNP  | B-NP   | B-ORG    |
| Phil           | NNP  | I-NP   | B-PER    |
| Simmons        | NNP  | I-NP   | I-PER    |

## 构建数据集

继承torch的Dataset父类。

```python
from torch.utils.data import Dataset, DataLoader

class Dataset(Dataset):
    def __init__(self, data, label, labelIndex, tokenizer, maxlength):
        self.data = data # 输入的原始数据
        self.label = label # 输入的原始数据的标签
        self.labelIndex = labelIndex # 标签字典
        self.tokernizer = tokenizer # 分类器
        self.maxlength = maxlength # 每个输入序列的最大长度（包括特殊标记）。

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
```

## 前向传播

在深度学习模型的训练和验证过程中，前向传播的调用方式确实会有所不同，主要是因为训练阶段和验证阶段的目的不同。以下是对这两行代码的详细解释：

```python
pre = model.forward(batchdata)
```

- **背景**：这行代码通常出现在验证或测试阶段。
- **参数**：只传入 `batchdata`。
- **目的**：在验证或测试阶段，我们只需要模型的预测结果，因此只需输入数据。模型将基于输入数据返回预测结果（`pre`）。
- **示例场景**：例如，我们在验证集上评估模型性能时，只需要前向传播计算预测值，而不需要计算损失。

```
loss = model.forward(batchdata, batchlabel)
```

- **背景**：这行代码通常出现在训练阶段。
- **参数**：同时传入 `batchdata` 和 `batchlabel`。
- **目的**：在训练阶段，除了输入数据外，我们还需要真实标签（`batchlabel`）来计算损失。模型将根据输入数据和真实标签进行前向传播，返回损失值（`loss`）。
- **示例场景**：在训练过程中，我们需要通过损失值来指导模型的学习和参数更新，因此必须提供标签。
