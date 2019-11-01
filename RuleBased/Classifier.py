import torch.nn as nn
import torch.optim as optim
from sklearn import datasets
import numpy as np
import torch

from RuleBased.Params import pca_or_cwa
from ALogger import ALogger


class LogisticRegression(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.out = nn.Linear(input_size, 1, bias=True)
        self.logger = ALogger("Classifier.py", True).getLogger()

    def forward(self, input):
        return torch.sigmoid(self.out(input))

    def update(self, train_feature_data, epoch, batch_size):

        my_criterion = nn.BCELoss()
        if pca_or_cwa == "pca":
            my_criterion = nn.MSELoss(size_average=True)
        my_optimizer = optim.Adam(self.parameters(), lr=0.001, weight_decay=0.001)

        for epoch_i in range(epoch):
            np.random.shuffle(train_feature_data)
            loss_running = 0
            start = 0
            while start + batch_size <= len(train_feature_data):
                self.zero_grad()
                tmp_data = train_feature_data[start:start + batch_size]
                train_x = tmp_data[:, 0:-1]
                train_y = tmp_data[:, -1]
                output = self.forward(torch.Tensor(train_x))
                loss = my_criterion(
                    torch.squeeze(output, 1), torch.Tensor(train_y))
                loss_running += loss.item()
                loss.backward()
                my_optimizer.step()
                start += batch_size

            if (epoch_i + 1) % 100 == 0 or epoch_i == epoch - 1:
                self.logger.info("Epoch:{} Loss:{}".format(epoch_i, loss_running))

                x = train_feature_data[:, 0:-1]
                y = train_feature_data[:, -1]
                y[np.where(y != 1)] = 0
                prec = self.test_precision(x, y)
                map = self.test_map(x, y)

                self.logger.info("Prec: {}, MAP: {}".format(prec, map))

    def test_precision(self, x, y):
        output = self.forward(torch.Tensor(x))
        output_label = (output.squeeze(-1).detach().numpy() > 0.5) * 1
        precision = np.sum((output_label == np.array(y)) * 1) / len(x)
        return precision

    def test_map(self, x, y):
        output = self.forward(torch.Tensor(x))
        output = output.squeeze(-1).detach().numpy()
        test = list(zip(output, y))
        test.sort(key=lambda p: p[0], reverse=True)
        MAP_metric = 0.0
        posi_num = 0
        for idx, a_pair in enumerate(test):
            if a_pair[1] == 1:
                posi_num += 1
                MAP_metric += (1.0 * posi_num) / (idx + 1)
        return MAP_metric / posi_num

    def get_output_prob(self, x):
        output = self.forward(torch.Tensor(x))
        prob = output.view(-1, 1).detach().numpy()[0]
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
    precision = np.sum(
        (output_label == np.array(y_data_list)) * 1) / len(test[seg:])
    print(precision)
