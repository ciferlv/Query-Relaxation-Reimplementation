from torch import nn
import torch


class PathRNN(nn.Module):
    def __init__(self, r_num, r_dim):
        super(PathRNN, self).__init__()
        self.r_embedding = nn.Embedding(num_embeddings=r_num, embedding_dim=r_dim)
        self.W_sigma = nn.Linear(in_features=r_dim * 2, out_features=r_dim, bias=True)
        self.init_weight()
        self.criterion = nn.MSELoss()
        self.zero_grad()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def init_weight(self):
        nn.init.xavier_uniform_(self.r_embedding.weight.data)
        nn.init.xavier_uniform_(self.W_sigma.weight.data)

    """
    Param:
    ------
    num: int, a single integer or list, a list of integer
    
    Returns:
    --------
    out: LongTensor([num]), a tensor
    """

    def int2tensor(self, num):
        if type(num) is type(1):
            return torch.LongTensor([num]).to(self.device)
        else:
            return torch.LongTensor(list(num)).to(self.device)

    def forward(self, input_path, tar_r_id):
        tar_r_embedding = self.r_embedding(self.int2tensor(tar_r_id))

        if len(input_path) == 2:
            cat_tensor = torch.cat((self.r_embedding(self.int2tensor(input_path[0])),
                                    self.r_embedding(self.int2tensor(input_path[1]))), -1)
            path_vec = torch.sigmoid(self.W_sigma(cat_tensor))
        elif len(input_path) == 1:
            path_vec = torch.sigmoid(self.r_embedding(self.int2tensor(input_path[0])))
        elif len(input_path) == 3:
            cat_tensor = torch.cat((self.r_embedding(self.int2tensor(input_path[0])),
                                    self.r_embedding(self.int2tensor(input_path[1]))), -1)
            path_vec = torch.sigmoid(self.W_sigma(cat_tensor))
            cat_tensor = torch.cat((path_vec, self.r_embedding(self.int2tensor(input_path[2]))), -1)
            path_vec = torch.sigmoid(self.W_sigma(cat_tensor))
        output = torch.sigmoid(torch.matmul(path_vec, torch.transpose(tar_r_embedding, 0, 1)))
        return output

    def get_loss(self, input_path, tar_r_id, label):
        output = self.forward(input_path, tar_r_id)
        loss = self.criterion(torch.squeeze(output,0), torch.FloatTensor([label]).to(self.device))
        return loss

    def get_most_similar_path(self, path_list, tar_r_id):
        simi_list = []
        for path in path_list:
            if len(path) == 1 and path[0] == tar_r_id:
                simi_list.append(-1)
                continue
            output = self.forward(path, tar_r_id)
            simi_list.append(output.cpu().detach().numpy()[0][0])
        most_similar_path_id = simi_list.index(max(simi_list))
        return path_list[most_similar_path_id]

    def get_map(self, input_path, tar_r_id, r_id_list):
        output = self.forward(input_path=input_path, tar_r_id=r_id_list)

        output = output.detach().numpy()[0]
        label_r_simi = output[r_id_list.index(tar_r_id)]

        cnt = 0
        for one_simi in output:
            if label_r_simi >= one_simi:
                continue
            else:
                cnt += 1
        precision = 1 / (cnt + 1)
        return precision

    def update(self, lr):
        optimizer = torch.optim.Adagrad(self.parameters(), lr=lr)
        optimizer.step()
        self.zero_grad()

    def saveModel(self, save_path):
        torch.save(self.state_dict(), save_path)

    def loadModel(self, load_path):
        self.load_state_dict(torch.load(load_path))
