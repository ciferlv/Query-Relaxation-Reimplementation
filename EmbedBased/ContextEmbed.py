import torch.nn as nn
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ContextEmbed(nn.Module):
    def __init__(self, embed_dim, e_num, r_num, margin):
        super(ContextEmbed, self).__init__()
        low = -6 / np.sqrt(embed_dim)
        high = 6 / np.sqrt(embed_dim)
        e_embed = np.random.uniform(low=low, high=high, size=(e_num, embed_dim))
        self.e_embed = nn.Embedding(e_num, embed_dim, norm_type=2, _weight=torch.Tensor(e_embed).to(device))
        r_embed = np.random.uniform(low=low, high=high, size=(r_num, embed_dim))
        self.r_embed = nn.Embedding(r_num, embed_dim, norm_type=2, _weight=torch.Tensor(r_embed).to(device))
        self.margin = margin

    def train(self, train_r, p_h, p_t, n_h, n_t):
        num = len(train_r)
        train_r = self.r_embed(torch.LongTensor(train_r).to(device))
        p_h = self.e_embed(torch.LongTensor(p_h).to(device))
        p_t = self.e_embed(torch.LongTensor(p_t).to(device))
        n_h = self.e_embed(torch.LongTensor(n_h).to(device))
        n_t = self.e_embed(torch.LongTensor(n_t).to(device))
        p_loss = torch.sum(torch.norm(p_h + train_r - p_t, 2, -1)) / num
        n_loss = torch.sum(torch.norm(n_h + train_r - n_t, 2, -1)) / num
        loss = p_loss - n_loss + self.margin
        return loss.to(device)
