import torch.nn as nn
import torch.optim as optim
from sklearn import datasets
import numpy as np
import torch
import math

from RuleBased.VirtuosoSearch.ALogger import ALogger


class LogisticRegression(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.out = nn.Linear(input_size, 1, bias=True)
        self.logger = ALogger("Classifier.py", True).getLogger()

    def forward(self, input):
        output = torch.sigmoid(self.out(input))
        return output

    def train(self, x, y, epoch, mini_batch):
        train_nums = len(x)
        mini_batch_nums = math.ceil(train_nums / mini_batch)
        seg_point_list = [0]
        for i in range(mini_batch_nums):
            if (i + 1) * mini_batch > train_nums:
                seg_point_list.append(train_nums)
            else:
                seg_point_list.append((i + 1) * mini_batch)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        for epoch_i in range(epoch):
            loss_running = 0
            for idx, seg_point in enumerate(seg_point_list):
                if idx == len(seg_point_list) - 1: break
                self.zero_grad()
                start_point = seg_point_list[idx]
                end_point = seg_point_list[idx + 1]
                train_x = x[start_point:end_point]
                train_y = y[start_point:end_point]
                output = self.forward(torch.Tensor(train_x))
                loss = criterion(torch.squeeze(output,1), torch.Tensor(train_y))
                loss_running += loss.item()
                loss.backward()
                optimizer.step()
            if (epoch_i % 100 == 0 or epoch_i == epoch -1) and epoch_i != 0:
                self.logger.info("Epoch:{} Loss:{}".format(epoch_i,loss_running))

    def test(self, x, y):
        output = self.forward(torch.Tensor(x))
        output_label = (output.squeeze(-1).detach().numpy() > 0.5) * 1
        precision = np.sum((output_label == np.array(y)) * 1) / len(x)
        # self.logger.info("Precision: {}".format(precision))
        return precision

    def get_output_prob(self,x):
        output = self.forward(torch.Tensor(x))
        prob = output.squeeze(-1).detach().numpy()[0]
        return prob

    def saveModel(self, savePath):
        torch.save(self.state_dict(), savePath)

    def loadModel(self, loadPath):
        self.load_state_dict(torch.load(loadPath))


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
