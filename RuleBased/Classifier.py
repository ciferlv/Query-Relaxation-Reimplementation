import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import datasets
import numpy as np
import torch
from torch.autograd import Variable


class LogisticRegression(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.out = nn.Linear(input_size, 1, bias=True)

    def forward(self, input):
        output = F.logsigmoid(self.out(input))
        return output

    def update(self, input, label):
        output = self.forward(input)
        loss = -F.reduce(output * label + (1 - label) * (1 - output))


if __name__ == "__main__":
    iris = datasets.load_iris()
    x = iris.data[:, :2]
    y = (iris.target != 0) * 1

    test = list(zip(x, y))
    np.random.shuffle(test)

    epoch = 10
    lg = LogisticRegression(2)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(lg.parameters())
    for i_epoch in range(epoch):
        # y_pred = lg(torch.Tensor(x))
        # loss = criterion(y_pred,torch.Tensor(y).unsqueeze(1))
        # loss.backward()
        # optimizer.step()
        x_data_list = []
        y_data_list = []
        for idx, data in enumerate(test[:40],start=1):
            x_data,y_data = data
            x_data_list.append(x_data)
            y_data_list.append([y_data])
            if idx % 10 == 0:
                output = lg(torch.Tensor(x_data_list))
                print(output)
                optimizer.zero_grad()
                loss = criterion(output,torch.Tensor(y_data_list))
                print(loss)
                loss.backward()
                optimizer.step()
                x_data_list = []
                y_data_list = []
    x_data_list = []
    y_data_list = []
    for data in test[40:]:
        x_data,y_data = data
        x_data_list.append(x_data)
        y_data_list.append([y_data])
    output = lg(torch.Tensor(x_data_list))
    print(output)
    # print(np.sum(output > 0.5 * 1))

