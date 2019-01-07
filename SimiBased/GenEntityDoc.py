from dit.divergences import kullback_leibler_divergence as kl_divergence
from dit.divergences import jensen_shannon_divergence as js_divergence
import dit
import numpy as np

from SimiBased.SimiBasedParams import kl_key_differ_num_threshold, alpha, kl_top_k


class SimiGraph:
    def __init__(self, triple2idx_file, e2idx_file, r2idx_file):
        self.triple2idx_file = triple2idx_file
        self.e2idx_file = e2idx_file
        self.r2idx_file = r2idx_file
        self.e_name2idx = {}
        self.e_idx2name = {}
        self.r_name2idx = {}
        self.r_idx2name = {}

        self.U_e_w_num = {}
        self.U_e_num = {}
        self.B_e_w_num = {}
        self.B_e_num = {}
        self.U_w_num = {}
        self.e_B_w_num = {}

        self.S_r_w_num = {}
        self.S_r_num = {}
        self.O_r_w_num = {}
        self.O_r_num = {}
        self.B_r_w_num = {}
        self.B_r_num = {}

        self.r_B_w_num = {}
        self.S_w_num = {}
        self.O_w_num = {}

        self.total_e_num = 0
        self.total_s_o_num = 0
        self.total_e_r_num = 0
        self.total_s_num = 0
        self.total_o_num = 0

    # w_token is e,r
    def count_w_of_e_B(self, w_token):
        if w_token not in self.e_B_w_num:
            self.e_B_w_num[w_token] = 0
        self.e_B_w_num[w_token] += 1

    # key_e is e_id, doc_e is w_id（e_id）
    def add_U_e_w(self, key_e, doc_e):
        assert type(key_e) == type(1), "key_e has wrong type."
        assert type(doc_e) == type(1), "doc_e has wrong type."

        if key_e not in self.U_e_w_num:
            self.U_e_w_num[key_e] = {}
        if doc_e not in self.U_e_w_num[key_e]:
            self.U_e_w_num[key_e][doc_e] = 0
        self.U_e_w_num[key_e][doc_e] += 1

        if key_e not in self.U_e_num:
            self.U_e_num[key_e] = 0
        self.U_e_num[key_e] += 1

    # key_e is e_id, doc_e_r is w_token
    def add_B_e_w(self, key_e, doc_e_r):
        assert type(key_e) == type(1), "key_e has wrong type."
        assert type(doc_e_r) == type("1"), "doc_e has wrong type."
        if key_e not in self.B_e_w_num:
            self.B_e_w_num[key_e] = {}
        if doc_e_r not in self.B_e_w_num[key_e]:
            self.B_e_w_num[key_e][doc_e_r] = 0
        self.B_e_w_num[key_e][doc_e_r] += 1

        if key_e not in self.B_e_num:
            self.B_e_num[key_e] = 0
        self.B_e_num[key_e] += 1

    # key_r is r_id, w_id is subject of r_id
    def add_S_r_w(self, key_r, w_id):
        assert type(key_r) == type(1), "key_r has wrong type."
        assert type(w_id) == type(1), "e_subject has wrong type."
        if key_r not in self.S_r_w_num:
            self.S_r_w_num[key_r] = {}
        if w_id not in self.S_r_w_num[key_r]:
            self.S_r_w_num[key_r][w_id] = 0
        self.S_r_w_num[key_r][w_id] += 1

        if key_r not in self.S_r_num:
            self.S_r_num[key_r] = 0
        self.S_r_num[key_r] += 1

    # key_r is r_id, w_id is object of r_id
    def add_O_r_w(self, key_r, w_id):
        assert type(key_r) == type(1), "key_r has wrong type."
        assert type(w_id) == type(1), "e_object has wrong type."
        if key_r not in self.O_r_w_num:
            self.O_r_w_num[key_r] = {}
        if w_id not in self.O_r_w_num[key_r]:
            self.O_r_w_num[key_r][w_id] = 0
        self.O_r_w_num[key_r][w_id] += 1

        if key_r not in self.O_r_num:
            self.O_r_num[key_r] = 0
        self.O_r_num[key_r] += 1

    # key_r is r_id, w_token is 'subject_id,object_id' of r_id
    def add_B_r_w(self, key_r, w_token):
        assert type(key_r) == type(1), "key_r has wrong type."
        assert type(w_token) == type("1"), "o_s has wrong type."
        if key_r not in self.B_r_w_num:
            self.B_r_w_num[key_r] = {}
        if w_token not in self.B_r_w_num[key_r]:
            self.B_r_w_num[key_r][w_token] = 0
        self.B_r_w_num[key_r][w_token] += 1

        if key_r not in self.B_r_num:
            self.B_r_num[key_r] = 0
        self.B_r_num[key_r] += 1

    # w_id is the id of e as subject or object
    def count_U_w_num(self, w_id):
        if w_id not in self.U_w_num:
            self.U_w_num[w_id] = 0
        self.U_w_num[w_id] += 11

    def load_triples(self):
        print("Start loading data.")
        with open(self.triple2idx_file, 'r', encoding="UTF-8") as f:
            for line in f.readlines():
                h, r, t = [int(w) for w in line.split()]
                self.total_e_num += 2
                self.total_e_r_num += 2
                self.total_s_o_num += 1
                self.total_s_num += 1
                self.total_o_num += 1

                self.count_U_w_num(h)
                self.count_U_w_num(t)
                self.count_w_of_e_B("{},{}".format(t, r))
                self.count_w_of_e_B("{},{}".format(h, r))

                self.add_U_e_w(h, t)
                self.add_U_e_w(t, h)
                self.add_B_e_w(h, "{},{}".format(t, r))
                self.add_B_e_w(t, "{},{}".format(h, r))
                self.add_S_r_w(r, h)
                self.add_O_r_w(r, t)
                self.add_B_r_w(r, "{},{}".format(h, t))

    def getLM(self):
        print("Start Calculating LM")

        def change_num2ratio(every_num_dict, total_num_dict):
            for key_id in every_num_dict:
                for w_id in every_num_dict[key_id]:
                    every_num_dict[key_id][w_id] = every_num_dict[key_id][w_id] / total_num_dict[key_id]

        change_num2ratio(self.U_e_w_num, self.U_e_num)
        change_num2ratio(self.B_e_w_num, self.B_e_num)

        change_num2ratio(self.S_r_w_num, self.S_r_num)
        change_num2ratio(self.O_r_w_num, self.O_r_num)
        change_num2ratio(self.B_r_w_num, self.B_r_num)

        def change_num2ratio_1(every_num_dict, total_num):
            for w_id in every_num_dict:
                every_num_dict[w_id] = every_num_dict[w_id] / total_num

        change_num2ratio_1(self.U_w_num, self.total_e_num)
        change_num2ratio_1(self.e_B_w_num, self.total_e_r_num)
        change_num2ratio_1(self.S_w_num, self.total_s_num)
        change_num2ratio_1(self.O_w_num, self.total_o_num)
        change_num2ratio_1(self.r_B_w_num, self.total_s_o_num)

    def get_js_value(self, p_dict, q_dict, prob_mode):
        key_set = p_dict.keys() | q_dict.keys()
        p_key_list = []
        p_prob_list = []
        q_key_list = []
        q_prob_list = []

        def add_one_prob(tar_key, prob_dict, tar_key_list, tar_prob_list):
            tar_key_list.append(tar_key)
            if tar_key in prob_dict:
                tar_prob_list.append(prob_dict[tar_key])
            else:
                if prob_mode == 0:  # entity Unigram
                    tar_prob_list.append(self.U_e_num[tar_key])
                if prob_mode == 1:  # entity Bigram
                    tar_prob_list.append(self.B_e_num[tar_key])
                if prob_mode == 2:  # relation Subject
                    tar_prob_list.append(self.S_r_num[tar_key])
                if prob_mode == 3:  # relation Object
                    tar_prob_list.append(self.O_r_num[tar_key])
                if prob_mode == 4:  # relation Bigram (Subject, Object)
                    tar_prob_list.append(self.B_r_num[tar_key])

        for key in key_set:
            add_one_prob(key, p_dict, p_key_list, p_prob_list)
            add_one_prob(key, q_dict, q_key_list, q_prob_list)

        p_prob_array = np.array(p_prob_list)
        q_prob_array = np.array(q_prob_list)

        p_prob_array = p_prob_array / p_prob_array.sum()
        q_prob_array = q_prob_array / q_prob_array.sum()

        p = dit.ScalarDistribution(p_key_list, p_prob_array)
        q = dit.ScalarDistribution(q_key_list, q_prob_array)
        return js_divergence(p, q)

    def load_e_r_dict(self):
        print("Load e2idx_file and r2idx_file.")
        with open(self.e2idx_file, 'r', encoding="UTF-8") as f:
            for line in f.readlines():
                idx, name = line.split()
                self.e_name2idx[name] = int(idx)
                self.e_idx2name[int(idx)] = name
        with open(self.r2idx_file, 'r', encoding="UTF-8") as f:
            for line in f.readlines():
                idx, name = line.split()
                if idx % 2 == 0:
                    self.r_name2idx[name] = int(idx)
                    self.r_idx2name[int(idx)] = name
        print("Finish loading e2idx_file and r2idx_file.")

    def get_top_K_simi_entity(self, p_e_name, k):
        print("Get top {} similar entities of entity: {}.".format(k, p_e_name))
        result_dict = {}
        p_e_idx = self.e_name2idx[p_e_name]
        U_p_dict = self.U_e_w_num[p_e_idx]
        B_p_dict = self.B_e_w_num[p_e_idx]
        for q_e_idx in self.e_idx2name:
            if q_e_idx == p_e_idx:
                continue
            U_q_dict = self.U_e_w_num[q_e_idx]
            if len(U_p_dict.keys()) - len(U_q_dict.keys()) >= kl_key_differ_num_threshold:
                break
            U_kl_value = self.get_js_value(U_p_dict, U_q_dict)

            B_q_dict = self.B_e_w_num[q_e_idx]
            if len(B_p_dict.keys()) - len(B_q_dict.keys()) >= kl_key_differ_num_threshold:
                break
            B_kl_value = self.get_js_value(B_p_dict, B_q_dict)
            result_dict[q_e_idx] = alpha * U_kl_value + (1 - alpha) * B_kl_value
        sorted_result_list = sorted(result_dict.items(), key=lambda kv: kv[1])
        sorted_result_list = sorted_result_list[:kl_top_k]
        for s_res in sorted_result_list:
            print("{}\t{}".format(self.e_idx2name[s_res[0]], self.s_res[1]))


def main():
    folder = "F:\\Data\\dbpedia\\All\\"

    e2idx_file = folder + "e2idx.txt"
    e2idx_shortcut_file = folder + "e2idx_shortcut.txt"

    # don't add inverse relation
    r2idx_file = folder + "r2idx.txt"

    # add inverse relation
    r2idx_shortcut_file = folder + "r2idx_shortcut.txt"

    triple2idx_file = folder + "triple2idx.txt"

    simiGraph = SimiGraph(triple2idx_file, e2idx_shortcut_file, r2idx_shortcut_file)
    simiGraph.load_triples()
    simiGraph.getLM()

    while True:
        my_input = input("Please input: ")
        if my_input == "1":
            simiGraph.get_name_simi(my_input)


if __name__ == "__main__":
    main()
