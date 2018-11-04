from ContextEmbed import ContextEmbed
from Triple import Node
import numpy as np
import torch.optim

e2idx_file = "./data/e2idx.txt"
r2idx_file = "./data/r2idx.txt"
triple2idx_file = "./data/triple2idx.txt"

e2idx = {}
idx2e = {}
r2idx = {}
idx2r = {}
node_dict = {}
head_context_dict = {}
tail_context_dict = {}


def load_data():
    with open(e2idx_file, 'r', encoding="UTF-8") as f:
        for line in f.readlines():
            idx, name = line.strip().split()
            e2idx[name] = int(idx)
            idx2e[int(idx)] = name
    with open(r2idx_file, 'r', encoding="UTF-8") as f:
        for line in f.readlines():
            idx, name = line.strip().split()
            r2idx[name] = int(idx)
            idx2r[idx] = name
    with open(triple2idx_file, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            h, r, t = line.strip().split()
            h = int(h)
            r = int(r)
            t = int(t)
            if h not in head_context_dict:
                head_context_dict[h] = Node(h)
            if t not in tail_context_dict:
                tail_context_dict[t] = Node(t)
            head_context_dict[h].addPath(r=r, e=t)
            tail_context_dict[t].addPath(r=r, e=h)


epoch = 200
minibatch_num = 2
e_num = 6000000
r_num = 700
embed_dim = 50
margin = 1
learning_rate = 0.001
head_context_threshold = 10
tail_context_threshold = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    ce = ContextEmbed(embed_dim=embed_dim, e_num=e_num, r_num=r_num, margin=margin).to(device)
    optimizer = torch.optim.Adam(ce.parameters(), lr=learning_rate)
    minibatch_size = int(len(idx2e.keys()) / 2)
    e_idx_list = list(idx2e.keys())
    max_idx = max(e_idx_list)
    for epoch_i in range(epoch):
        for minibatch_i in range(minibatch_num):
            ce.zero_grad()
            start = minibatch_i * minibatch_size
            end = (minibatch_i + 1) * minibatch_size
            if end > len(e_idx_list): end = len(e_idx_list)
            batch_idx_list = e_idx_list[start:end]
            running_loss = 0
            for current_idx in batch_idx_list:
                print("{}".format(current_idx))
                head_context = None
                tail_context = None
                if current_idx in head_context_dict:
                    head_context = head_context_dict[current_idx]
                    head_context.sample_path(head_context_threshold)

                if current_idx in tail_context_dict:
                    tail_context = tail_context_dict[current_idx]
                    tail_context.sample_path(tail_context_threshold)

                while True:
                    nege_node_idx = np.random.randint(0, max_idx)
                    if nege_node_idx not in idx2e or nege_node_idx == current_idx: continue

                    if head_context is not None and nege_node_idx in head_context_dict:
                        nege_head_context = head_context_dict[nege_node_idx]
                        if head_context.has_intersection(nege_head_context.path_list): continue

                    if tail_context is not None and nege_node_idx in tail_context_dict:
                        nege_tail_context = tail_context_dict[nege_node_idx]
                        if tail_context.has_intersection(nege_tail_context.path_list): continue
                    break

                train_r, train_p_h, train_p_t, train_n_h, train_n_t = gen_train_data(head_context, tail_context,
                                                                                     current_idx,
                                                                                     nege_node_idx)
                loss = ce.train(train_r, train_p_h, train_p_t, train_n_h, train_n_t)
                if loss.item() > 0:
                    loss.backward()
                running_loss += loss.item()

            print("Epoch: {} MiniBatch: {} Loss: {}\n".format(epoch_i, minibatch_i, running_loss))
            optimizer.step()

        if (epoch_i + 1) % 20 == 0:
            filePath = "./data/model/model_epoch_" + str(epoch_i + 1) + ".tar"
            torch.save(ce.state_dict(), filePath)


def gen_train_data(posi_head_c, posi_tail_c, current_idx, nege_idx):
    posi_h = []
    posi_t = []
    nege_h = []
    nege_t = []
    r = []

    if posi_head_c is not None:
        r_list, tail_list = posi_head_c.gen_train_data()
        r.extend(r_list)
        posi_t.extend(tail_list)
        posi_h.extend(list(np.ones_like(r_list) * current_idx))

        nege_t.extend(tail_list)
        nege_h.extend(list(np.ones_like(r_list) * nege_idx))

    if posi_tail_c is not None:
        r_list, head_list = posi_tail_c.gen_train_data()
        r.extend(r_list)
        posi_h.extend(head_list)
        posi_t.extend(list(np.ones_like(r_list) * current_idx))

        nege_h.extend(head_list)
        nege_t.extend(list(np.ones_like(r_list) * nege_idx))

    return r, posi_h, posi_t, nege_h, nege_t


if __name__ == "__main__":
    load_data()
    train()
