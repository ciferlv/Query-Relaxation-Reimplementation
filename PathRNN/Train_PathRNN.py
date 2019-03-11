from Param import r_dim, r_num, epoch, mini_batch, decay_epoch
from PathRNN import PathRNN
from RuleBased.BiSearch.Graph import Graph
import math
import numpy as np
import os
import random
import torch


class TPRNN:
    def __init__(self):
        self.root_folder = "../Data/dbpedia/"
        # self.train_scope = "United_States"
        self.train_scope = "Canada"
        self.saved_model_folder = "./model/"
        self.saved_path_folder = "./path/"
        self.e2idx_file = self.root_folder + self.train_scope + "/e2idx_shortcut.txt"
        self.r2idx_file = self.root_folder + self.train_scope + "/r2idx_shortcut.txt"
        self.triple2idx_file = self.root_folder + self.train_scope + "/triple2idx.txt"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.posi_ht_list = []
        self.nege_ht_list = []

        self.posi_train_ht_list = []
        self.posi_test_ht_list = []
        self.nege_train_ht_list = []

        self.posi_train_ht2path_dict = []
        self.nege_train_ht2path_dict = []
        self.posi_test_ht2path_dict = []

        self.train_data = {}
        self.train_label = {}

        self.test_data = {}
        self.test_label = {}

    def load_data(self):
        self.graph = Graph(self.e2idx_file, self.r2idx_file, self.triple2idx_file)
        self.graph.load_data()

    def sample_posi_nege(self, tar_r_name):
        tar_r_idx = self.graph.r2idx[tar_r_name]
        print("Start sample posi and nege for R_name: {}, R_idx: {}.".format(tar_r_name, tar_r_idx))

        self.posi_ht_list = self.graph.r2ht[tar_r_idx]
        print("Num of posi_list is {}.".format(len(self.posi_ht_list)))

        num_per_time = math.ceil(len(self.posi_ht_list) / (len(self.graph.r2ht.keys()) - 1))
        print("Num to sample per time: {}.".format(num_per_time))

        for sample_r_idx in self.graph.r2ht:
            nege_len = len(self.nege_ht_list)
            if sample_r_idx != tar_r_idx:
                sample_ht = self.graph.r2ht[sample_r_idx]
                if num_per_time <= len(sample_ht):
                    self.nege_ht_list.extend(random.sample(sample_ht, num_per_time))
                else:
                    self.nege_ht_list.extend(sample_ht)
            print(
                "Sample {}/{} neges for R_name: {}, R_idx: {}.".format(len(self.nege_ht_list) - nege_len, num_per_time,
                                                                       self.graph.idx2r[sample_r_idx], sample_r_idx))

        print("Num of nege_list/posi_list is {}/{}.".format(len(self.nege_ht_list), len(self.posi_ht_list)))

        self.posi_train_ht_list = self.posi_ht_list[:math.floor(len(self.posi_ht_list) * 0.9)]
        self.posi_test_ht_list = self.posi_ht_list[math.floor(len(self.posi_ht_list) * 0.9):]

        self.nege_train_ht_list = self.nege_ht_list[:math.floor(len(self.nege_ht_list) * 0.9)]

        print("posi_train_ht: {}, posi_test_ht: {}".format(len(self.posi_train_ht_list),
                                                           len(self.posi_test_ht_list)))
        print("nege_train_ht: {}".format(len(self.nege_train_ht_list)))

    def search_path(self, tar_r_name):
        print("Search path for posi and nege.")
        path_saved_folder4r_name = self.saved_path_folder + tar_r_name.split(":")[-1] + "/"
        if not os.path.exists(path_saved_folder4r_name):
            os.makedirs(path_saved_folder4r_name)

        def record_path2file(file_name, ht_list):
            with open(file_name, 'w', encoding="UTF-8") as f:
                for idx, ht in enumerate(ht_list):
                    print("Collecting path of ht {}/{}.".format(idx, len(ht_list)))
                    h = ht[0]
                    t = ht[1]
                    record = "{} {};".format(h, t)
                    e_r_path_found_list = self.graph.search_bidirect(head=h, tail=t, step=3)
                    for e_r_path in e_r_path_found_list:
                        r_path = self.graph.extract_r_path(e_r_path)
                        record += "{};".format(" ".join([str(r_idx) for r_idx in r_path]))
                    f.write("{}\n".format(record.strip(";")))

        print("Record path of ht in posi_train")
        record_path2file(path_saved_folder4r_name + "posi_train.txt", self.posi_train_ht_list)

        print("Record path of ht in posi_test")
        record_path2file(path_saved_folder4r_name + "posi_test.txt", self.posi_test_ht_list)

        print("Record path of ht in nege_train")
        record_path2file(path_saved_folder4r_name + "nege_train.txt", self.nege_train_ht_list)

    def load_path_from_file(self, tar_r_name):
        print("Start loading path from graph.")
        path_saved_folder4r_name = self.saved_path_folder + tar_r_name.split(":")[-1] + "/"

        def get_path(file_name):
            ht_path_dict = {}
            with open(file_name, 'r', encoding="UTF-8") as f:
                for line in f.readlines():
                    path_array = line.split(";")
                    ht_path_dict[path_array[0]] = []
                    for path_str in path_array[1:]:
                        ht_path_dict[path_array[0]].append([int(r_idx) for r_idx in path_str.split()])
            return ht_path_dict

        print("Loading posi train ht2path.")
        self.posi_train_ht2path_dict = get_path(path_saved_folder4r_name + "posi_train.txt")
        for ht_key in self.posi_train_ht2path_dict:
            self.train_label[ht_key] = 1

        print("Loading posi test ht2path.")
        self.posi_test_ht2path_dict = get_path(path_saved_folder4r_name + "posi_test.txt")

        print("Loading nege train ht2path.")
        self.nege_train_ht2path_dict = get_path(path_saved_folder4r_name + "nege_train.txt")
        for ht_key in self.nege_train_ht2path_dict:
            self.train_label[ht_key] = 0

        self.test_data = self.posi_test_ht2path_dict
        self.train_data = self.posi_train_ht2path_dict.copy()
        self.train_data.update(self.nege_train_ht2path_dict)

    def train_model(self, tar_r_name):
        print("Start Training. Epoch: {}, Mini_Batch: {}, r_dim: {}, r_num: {}".format(epoch, mini_batch, r_dim, r_num))
        initial_learning_rate = 0.001
        tar_r_id = self.graph.r2idx[tar_r_name]
        p_rnn = PathRNN(r_dim=r_dim, r_num=r_num)
        train_ht_key_list = list(self.train_data.keys())
        for i_epoch in range(epoch):
            batch_cnt = math.ceil(len(self.train_data) / mini_batch)
            for i_batch in range(batch_cnt):
                start = i_batch * mini_batch
                end = (i_batch + 1) * mini_batch
                if end > len(self.train_data):
                    end = len(self.train_data)
                running_loss = 0
                for i_key in range(start, end):
                    ht_key = train_ht_key_list[i_key]
                    path = p_rnn.get_most_similar_path(self.train_data[ht_key], tar_r_id)
                    loss = p_rnn.get_loss(path, tar_r_id, self.train_label[ht_key])
                    running_loss += loss.item()
                    loss.backward()
                    # print("I_key: {}, Start: {}, End: {}, Total: {}".format(i_key, start, end, len(self.train_data)))
                print("Epoch: {}/{}, Batch: {}/{}, Loss: {}.".format(i_epoch, epoch, i_batch, batch_cnt,
                                                                     running_loss / (end - start)))
                p_rnn.update(lr=initial_learning_rate)
            if (i_epoch + 1) % 5 == 0:
                print("Save model.")
                save_model_folder = self.saved_model_folder + tar_r_name.split(":")[-1] + "/"
                if not os.path.exists(save_model_folder):
                    os.mkdir(save_model_folder)
                p_rnn.saveModel(save_model_folder + "model_{}.tar".format(i_epoch + 1))

                print("Epoch: {}".format(i_epoch))
                self.test_model(tar_r_name)

            if (i_epoch + 1) % decay_epoch == 0:
                initial_learning_rate /= 2

    def test_model(self, tar_r_name):
        tar_r_id = self.graph.r2idx[tar_r_name]
        print("Start testing model.")
        save_model_folder = self.saved_model_folder + tar_r_name.split(":")[-1] + "/"
        file_name_list = os.listdir(save_model_folder)
        file_name_list.sort(reverse=True)
        model_file = save_model_folder + file_name_list[0]
        p_rnn = PathRNN(r_dim=r_dim, r_num=r_num)
        p_rnn.loadModel(model_file)
        cnt_prec = 0
        for test_ht_key in self.test_data:
            path = p_rnn.get_most_similar_path(self.test_data[test_ht_key], tar_r_id)
            print("Test_Path: {}".format("->".join(self.graph.display_r_path([path])[0])))
            cnt_prec += p_rnn.get_map(input_path=path, tar_r_id=tar_r_id, r_id_list=list(self.graph.idx2r.keys()))
            print("{}".format(cnt_prec))
        print("Test Result, MAP: {}.".format(cnt_prec / len(self.test_data)))


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    tprnn = TPRNN()
    tar_r_name = "dbo:location"
    tprnn.load_data()
    # tprnn.sample_posi_nege(tar_r_name=tar_r_name)
    # tprnn.search_path(tar_r_name=tar_r_name)
    tprnn.load_path_from_file(tar_r_name)
    # tprnn.train_model(tar_r_name)
    tprnn.test_model(tar_r_name)
