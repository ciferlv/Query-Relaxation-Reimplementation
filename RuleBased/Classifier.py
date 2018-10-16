import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import datasets
import numpy as np
import torch


class LogisticRegression(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.out = nn.Linear(input_size, 1, bias=True)

    def forward(self, input):
        output = F.sigmoid(self.out(input))
        return output


if __name__ == "__main__":
    iris = datasets.load_iris()
    x = iris.data[:, :2]
    y = (iris.target != 0) * 1

    one_num = np.sum(y)
    zero_num = len(y) - one_num

    one_ratio = 1
    zero_ration = one_num / zero_num

    test = list(zip(x, y))
    np.random.shuffle(test)
    seg = 120
    epoch = 1000
    lg = LogisticRegression(2)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(lg.parameters(), lr=0.001)
    for i_epoch in range(epoch):
        x_data_list = []
        y_data_list = []
        ration_list = []
        for idx, data in enumerate(test[:seg], start=1):
            x_data, y_data = data
            x_data_list.append(x_data)
            y_data_list.append(y_data)

            if y_data == 1:
                ration_list.append(1)
            else:
                ration_list.append(2)

            if idx % 10 == 0:
                output = lg(torch.Tensor(x_data_list))
                optimizer.zero_grad()
                criterion.weight = torch.Tensor(ration_list)
                loss = criterion(output, torch.Tensor(y_data_list))
                loss.backward()
                optimizer.step()
                x_data_list = []
                y_data_list = []
                ration_list = []
    x_data_list = []
    y_data_list = []
    for data in test[seg:]:
        x_data, y_data = data
        x_data_list.append(x_data)
        y_data_list.append(y_data)
    output = lg(torch.Tensor(x_data_list))
    output_label = (output.squeeze(-1).detach().numpy() > 0.5) * 1
    print(output_label)
    print(y_data_list)
    precision = np.sum((output_label == np.array(y_data_list)) * 1) / len(test[seg:])
    print(precision)
