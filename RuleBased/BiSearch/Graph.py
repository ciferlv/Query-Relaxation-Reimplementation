from RuleBased.BiSearch.Triple import Node, Rule
import random
import numpy as np
import os

from RuleBased.Classifier import LogisticRegression
from RuleBased.Params import rule_seg, mydb


class Graph:
    def __init__(self, e2idx_file, r2idx_file, triple2idx_file):
        self.e2idx_file = e2idx_file
        self.r2idx_file = r2idx_file
        self.triple2idx_file = triple2idx_file

        self.e2idx = {}
        self.idx2e = {}
        self.r2idx = {}
        self.idx2r = {}
        self.node_dict = {}
        self.r2ht = {}

    def load_data(self):
        with open(self.e2idx_file, 'r', encoding="UTF-8") as f:
            for line in f.readlines():
                idx, name = line.strip().split()
                idx = int(idx)
                self.e2idx[name] = idx
                self.idx2e[idx] = name
        with open(self.r2idx_file, 'r', encoding="UTF-8") as f:
            for line in f.readlines():
                idx, name = line.strip().split()
                idx = int(idx)
                self.r2idx[name] = idx
                self.idx2r[idx] = name
                self.r2ht[idx] = []
        with open(self.triple2idx_file, 'r', encoding='UTF-8') as f:
            for line in f.readlines():
                h, r, t = line.strip().split()
                h = int(h)
                r = int(r)
                t = int(t)
                self.r2ht[r].append([h, t])
                if h not in self.node_dict:
                    self.node_dict[h] = Node(h)
                if t not in self.node_dict:
                    self.node_dict[t] = Node(t)
                self.node_dict[h].addPath(r=r, e=t)
                inv_r = "inv_{}".format(self.idx2r[r])
                self.node_dict[t].addPath(r=self.r2idx[inv_r], e=h)

    def find_intersection(self, left_path, right_path):
        res = []
        left_dict = {}
        right_dict = {}
        for l_p in left_path:
            end_point = l_p[-1]
            if end_point not in left_dict:
                left_dict[end_point] = []
            left_dict[end_point].append(l_p)
        for r_p in right_path:
            end_point = r_p[-1]
            if end_point not in right_dict:
                right_dict[end_point] = []
            right_dict[end_point].append(r_p)
        for key in left_dict.keys():
            if key in right_dict:
                for l_p in left_dict[key]:
                    for r_p in right_dict[key]:
                        temp_l = l_p[:-1]
                        temp_r = r_p.copy()
                        temp_r.reverse()
                        temp_l.extend(temp_r)
                        res.append(temp_l)
        return res

    def search_bidirect(self, head, tail, step):
        res = []
        left_path = self.search_unidirect(head, int(step / 2))
        right_path = self.search_unidirect(tail, step - int(step / 2))
        for step_i in range(step):
            left_len = int((step_i + 1) / 2)
            right_len = step_i - left_len
            temp_res = self.find_intersection(left_path[left_len], right_path[right_len])
            res.extend(temp_res)
        return res

    def search_unidirect(self, head, step):
        res = []
        res.append([[-1, head]])
        current_node_list = [[-1, head]]
        for i in range(step):
            temp_path = []
            for n in current_node_list:
                c_node = self.node_dict[n[-1]]
                for path in c_node.path_list:
                    if i >= 1 and path.e == n[-3]: continue
                    temp_n = n.copy()
                    temp_n.append(path.r)
                    temp_n.append(path.e)
                    temp_path.append(temp_n)
            res.append(temp_path.copy())
            current_node_list = temp_path.copy()
        return res

    def display_e_r_path(self, path_list):
        displayed_path = []
        for path in path_list:
            temp_display = []
            for idx in range(1, len(path) - 1):
                if idx % 2 != 0:
                    temp_display.append(self.idx2e[path[idx]])
                else:
                    temp_display.append(self.idx2r[path[idx]])
            displayed_path.append(temp_display)
        return displayed_path

    def display_r_path(self, r_path_list):
        displayed_path = []
        for r_path in r_path_list:
            temp = [self.idx2r[r_idx] for r_idx in r_path]
            displayed_path.append(temp)
        return displayed_path

    def extract_r_path(self, path):
        r_path = []
        for idx in range(1, len(path) - 1):
            if idx % 2 == 0:
                r_path.append(path[idx])
        return r_path

    def search_path(self, r_idx, max_step):
        searched_r_path = {}
        searched_e_r_path = []
        ht_list = self.r2ht[r_idx]
        print("Start Searching Path:\nR:{} Train Num:{}".format(self.idx2r[r_idx], int(0.1 * len(ht_list))))
        for ht in random.sample(ht_list, int(0.1 * len(ht_list))):
            h = ht[0]
            t = ht[1]
            path_found = self.search_bidirect(h, t, max_step)
            searched_e_r_path.extend(path_found)
            for p in path_found:
                r_path = self.extract_r_path(p)
                r_path_key = rule_seg.join(map(str, r_path))
                if r_path_key not in searched_r_path:
                    searched_r_path[r_path_key] = r_path
        return list(searched_r_path.values()), searched_e_r_path

    def get_passed_ht(self, r_path):
        ht_list = self.r2ht[r_path[0]]
        for idx in range(1, len(r_path)):
            temp_path = []
            r_idx = r_path[idx]
            for ht in ht_list:
                t = ht[-1]
                c_node = self.node_dict[t]
                tail_list = c_node.get_tails(r_idx)
                if tail_list is None:
                    continue
                else:
                    for tail in tail_list:
                        c_path = ht.copy()
                        c_path.append(tail)
                        temp_path.append(c_path)
            ht_list = temp_path
        return [[ht[0], ht[-1]] for ht in ht_list]

    def enhance_rule(self, r_idx, r_path_list):
        rule_list = []
        for r_path in r_path_list:
            rule = Rule(r_idx, r_path)
            succ = rule.restoreFromMysql()
            if not succ:
                rule.passHT = self.get_passed_ht(r_path)
                rule.get_P_R_F1(self.node_dict, self.r2ht)
                rule_list.append(rule)
                rule.persist2mysql()
        return rule_list

    """
    Train or load a model for relation: r_idx
    Parameters
    ----------
    r_idx: the index of a relation
    max_step: max steps for the rules
    top_rules_num: the rule num used to train model
    folder: the folder under which thd output model is saved
    
    Returns:
    ----------
    out: A trained model of relation r_idx
    
    """

    def get_r_model(self, r_idx, max_step, top_rules_num, folder):
        statistics_file = folder + "/statistics.txt"
        model_file_path = folder + "/model.tar"
        if os.path.exists(statistics_file) and os.path.exists(model_file_path):
            print("Load Model for Relation: {} Max Steps: {}".format(r_idx, max_step))
            with open(statistics_file, 'r', encoding="UTF-8") as f:
                input_size = int(f.readline().strip().split()[1])
            lg = LogisticRegression(input_size)
            lg.loadModel(model_file_path)
            return lg

        r_path, e_r_path = self.search_path(r_idx, max_step)
        rule_list = self.enhance_rule(r_idx, r_path)
        rule_list.sort(key=lambda one_rule: one_rule.P, reverse=True)
        if len(rule_list) <= top_rules_num:
            top_rules_num = len(rule_list)

        with open(statistics_file, 'w', encoding="UTF-8") as f:
            f.write("input_size\t{}\n".format(top_rules_num))

        posi_list = []
        nege_list = []

        for rule in rule_list[:top_rules_num]:
            posi, nege = rule.sample_train_data(100, 100)
            posi_list.extend(posi)
            nege_list.extend(nege)

        posi_features = []
        posi_labels = list(np.ones(len(posi_features)))
        nege_features = []
        nege_labels = list(np.zeros(len(nege_features)))
        for posi_ht in posi_list:
            feature = self.get_features(rule_list[:top_rules_num], posi_ht)
            posi_features.append(feature)
        for nege_ht in nege_list:
            feature = self.get_features(rule_list[:top_rules_num], nege_ht)
            nege_features.append(feature)

        train_x = posi_features.append(nege_features)
        train_y = posi_labels.append(nege_labels)
        print("Start training model for r:{} max_steps:{} top_rules_num:{}".format(r_idx, max_step, top_rules_num))

        lg = LogisticRegression(top_rules_num)
        lg.train(train_x, train_y, 500, 500)
        lg.saveModel(model_file_path)
        return lg

    """
    Get features for one pair of h and t
    Parameters:
    -----------
    rule_list: list
    It stores a list of rules.
    ht: list
    a list of length 2, for example, [head,tail]
    
    Returns:
    -----------
    out: list
    A list of features, every entry represents if a rule is passed.
    """

    def get_features(self, rule_list, ht):
        feature = []
        for rule in rule_list:
            if rule.is_correct_ht(ht):
                feature.append(1)
            else:
                feature.append(0)
        return feature

    def get_top_k_rules(self, r_idx, top_k, criterion):
        rule_list = []
        query = "select relation_idx, rule_key from dbpediarule" \
                " where relation_idx={} order by {} desc".format(r_idx, criterion)
        mycursor = mydb.cursor()
        mycursor.execute(query)
        fetched = mycursor.fetchall()
        for idx, row in enumerate(fetched):
            if idx > top_k - 1: break
            one_rule = Rule(r_idx, list(map(int, row[2].split(':'))))
            one_rule.restoreFromMysql()
            rule_list.append(one_rule)
        return rule_list


if __name__ == "__main__":
    e2idx_file = "./data/FB15K-237/e2idx.txt"
    r2idx_file = "./data/FB15K-237/r2idx.txt"
    triple2idx_file = "./data/FB15K-237/triple2idx.txt"
    graph = Graph(e2idx_file, r2idx_file, triple2idx_file)
    graph.load_data()
    r_path_list, e_r_path_list = graph.search_path(2, 3)

    # displayed_path = graph.display_r_path(r_path_list)
    # displayed_path = graph.display_e_r_path(e_r_path_list)
    # for p in displayed_path:
    #     print("=>".join(p))
