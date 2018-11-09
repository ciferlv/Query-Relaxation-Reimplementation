from RuleBased.BiSearch.Triple import Node, Rule
import random
import numpy as np
import os

from RuleBased.Classifier import LogisticRegression
from RuleBased.Params import rule_seg, mydb, file_path_seg, database, ht_conn, ht_seg, sampled_num_to_search_rule, \
    top_frequency_rule_num
from RuleBased.Util import Util


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
        print("Start Loading Graph")
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
                self.r2ht[self.r2idx[inv_r]].append([t, h])
                self.node_dict[t].addPath(r=self.r2idx[inv_r], e=h)
        print("Finishing Loading Graph")

    '''
    Connect two path found by unidirect search
    Parameters:
    -----------
    left_path: list [[-1,e_idx,r_idx,...],[]]
    right_path: list [[-1,e_idx,r_idx,...],[],...]
    
    Returns:
    -----------
    out: list [[-1,e_idx,r_idx,...,e_idx,-1],[],[],...]
    '''

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
                        for r_p_i in range(len(r_p) - 1, 0, -1):
                            if r_p_i % 2 != 0:
                                temp_l.append(r_p[r_p_i])
                            else:
                                name = self.idx2r[r_p[r_p_i]]
                                if name.startswith("inv_"):
                                    name = name[4:]
                                else:
                                    name = "inv_" + name
                                temp_l.append(self.r2idx[name])
                        temp_l.append(-1)
                        res.append(temp_l)
        return res

    """
    Search from bidirection
    Parameters:
    -----------
    head: int 
    index of head entity
    tail: int
    index of tail entity
    step: int
    the max length between head and tail.
    for example, when step = 3, we get path whose length is 1,2 and 3
    
    Returns:
    -----------
    out: list [[-1,e_idx,r_idx,e_idx,...,-1],[],[],...]
    """

    def search_bidirect(self, head, tail, step):
        res = []
        left_path = self.search_unidirect(head, int(step / 2))
        right_path = self.search_unidirect(tail, step - int(step / 2))
        for step_i in range(step):
            left_len = int((step_i + 1) / 2)
            right_len = (step_i + 1) - left_len
            temp_res = self.find_intersection(left_path[left_len], right_path[right_len])
            res.extend(temp_res)
        return res

    """
    Search from head to get path whose length is under step
    for example, when step is 3, the length of path got is 1,2,3
    Parameters:
    -----------
    head: int (e_idx)
    index of start point
    step: int
    max length of path
    
    Returns:
    -----------
    out:  list [[-1,e_idx,...],[],...]
    The list of path for searched.
    """

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

    """
    Convert r_idx to r_name, e_idx to e_name:
    Parameters:
    -----------
    e_r_path_list: list [[-1,e_idx,r_idx,e_idx,..,-1],[],...]
    
    Returns:
    out: list [[e_name,r_name,e_name,...],[],...]
    """

    def display_e_r_path(self, e_r_path_list):
        displayed_path = []
        for path in e_r_path_list:
            temp_display = []
            for idx in range(1, len(path) - 1):
                if idx % 2 != 0:
                    temp_display.append(self.idx2e[path[idx]])
                else:
                    temp_display.append(self.idx2r[path[idx]])
            displayed_path.append(temp_display)
        return displayed_path

    """
    Convert r_idx to r_name
    Parameters:
    -----------
    r_path_list: list, [[r_idx,r_idx,...],[],...]
    
    Returns:
    -----------
    out: list, [[r_name,r_name,...],[r_name,r_name,...],...]
    """

    def display_r_path(self, r_path_list):
        displayed_path = []
        for r_path in r_path_list:
            temp = [self.idx2r[r_idx] for r_idx in r_path]
            displayed_path.append(temp)
        return displayed_path

    """
    Extract r_path for e_r_path
    Parameters:
    -----------
    e_r_path: list [-1,e_idx,r_idx,e_idx,...,-1]
    
    Returns:
    -----------
    out: list [r_idx,r_idx,...]
    """

    def extract_r_path(self, e_r_path):
        r_path = []
        for idx in range(1, len(e_r_path) - 1):
            if idx % 2 == 0:
                r_path.append(e_r_path[idx])
        return r_path

    """
    Search paths that can conclude to r_idx
    Parameters:
    -----------
    r_idx: int 
    index of target relation(r_idx)
    max_step: int 
    max length of searched path
    
    Returns:
    -----------
    out: list, list
    first list is the path of r, [r_idx,r_idx,r_idx,...]
    second list is the path of e and r, [-1,e_idx,r_idx,e_idx,r_iex,...,-1]
    """

    def search_path(self, r_idx, max_step):
        search_r_path_num = {}
        searched_r_path = {}
        searched_e_r_path = []
        ht_list = self.r2ht[r_idx]
        sampled_num = int(0.1 * len(ht_list))
        if int(0.1 * len(ht_list)) > sampled_num_to_search_rule:
            sampled_num = sampled_num_to_search_rule
        print("Start Searching Path:\nR:{} Train Num:{}".format(self.idx2r[r_idx], sampled_num))
        for idx, ht in enumerate(random.sample(ht_list, sampled_num)):
            h = ht[0]
            t = ht[1]
            print('{}/{} Relation: {} H: {} T:{}'.format(idx + 1, sampled_num, self.idx2r[r_idx], self.idx2e[h],
                                                         self.idx2e[t]))
            path_found = self.search_bidirect(h, t, max_step)
            for p in path_found:
                r_path = self.extract_r_path(p)
                r_path_key = rule_seg.join(map(str, r_path))
                if len(r_path) == 1 and r_path[0] == r_idx: continue
                searched_e_r_path.append(p)
                if r_path_key not in searched_r_path:
                    searched_r_path[r_path_key] = r_path
                    search_r_path_num[r_path_key] = 1
                else:
                    search_r_path_num[r_path_key] += 1
        res_r_path_list = []
        for key, value in list(sorted(search_r_path_num.items(), key=lambda d: d[1], reverse=True))[
                          :top_frequency_rule_num]:
            res_r_path_list.append(searched_r_path[key])
        return res_r_path_list, searched_e_r_path

    '''
    Load rules for relation(r_idx), the length of rules is under max_step
    Parameters:
    -----------
    r_idx: int, index of relation
    max_step: int, max length of rule of relation(r_idx)
    
    Returns:
    -----------
    out: list, the list of rules for relation(r_idx) loaded from mysql
    '''

    def load_rule_from_mysql(self, r_idx, max_step):
        rule_list = []
        sql = "select * from " + database + " where relation_idx = {} and rule_len <= {};".format(r_idx, max_step)
        mycursor = mydb.cursor()
        mycursor.execute(sql)
        fetched = mycursor.fetchall()
        for row in fetched:
            rule = Rule(r_idx, None, row[2])
            rule.rule_len = int(row[3])
            rule.correct_ht = [list(map(int, ht2)) for ht2 in [ht.split(ht_conn) for ht in row[4].split(ht_seg)]]
            rule.wrong_ht = [list(map(int, ht2)) for ht2 in [ht.split(ht_conn) for ht in row[5].split(ht_seg)]]
            rule.no_idea_ht = [list(map(int, ht2)) for ht2 in [ht.split(ht_conn) for ht in row[6].split(ht_seg)]]
            rule.P = row[7]
            rule.R = row[8]
            rule.F1 = row[9]
            rule_list.append(rule)
        return rule_list

    '''
    Convert relation(r_idx) to its inverse relation(r_idx)
    Parameters:
    -----------
    r_idx: int 
    indxe of relation
    
    Returns:
    -----------
    out: int
    index of its inverse relation
    '''

    def convert_r(self, r_idx):
        name = self.idx2r[r_idx]
        if name.startswith("inv_"):
            name = name[4:]
        else:
            name = "inv_" + name
        return self.r2idx[name]

    """
    Get a list of ht which can pass r_path
    Parameters:
    -----------
    r_path: list 
    a rule: [r_idx,r_idx,r_idx,...], for example: [2,3,6,...]
    
    Returns:
    out: list
    a list of ht which can pass this r_path
    [[h,t],[h,t],[h,t],[h,t],...]
    """

    def get_passed_ht(self, r_path):
        if len(r_path) == 1:
            return self.r2ht[r_path[0]]

        left_path = self.r2ht[r_path[0]]
        left_step = 1
        inv_r_idx = self.convert_r(r_path[-1])
        right_path = self.r2ht[inv_r_idx]
        right_step = 1
        while len(r_path) - (left_step + right_step) > 0:
            temp_left_path = []
            temp_right_path = []
            if len(left_path) < len(right_path):
                left_step += 1
                r_idx = r_path[left_step - 1]
                for ht in left_path:
                    c_node = self.node_dict[ht[-1]]
                    for tail in c_node.get_tails(r_idx):
                        temp_ht = ht.copy()
                        temp_ht.append(tail)
                        temp_left_path.append(temp_ht)
                left_path = temp_left_path
            else:
                right_step += 1
                r_idx = r_path[-right_step]
                inv_r_idx = self.convert_r(r_idx)
                for ht in right_path:
                    c_node = self.node_dict[ht[-1]]
                    for tail in c_node.get_tails(inv_r_idx):
                        temp_ht = ht.copy()
                        temp_ht.append(tail)
                        temp_right_path.append(temp_ht)
                right_path = temp_right_path
        res = set()
        left_dict = {}
        for path in left_path:
            if path[-1] not in left_dict:
                left_dict[path[-1]] = []
            left_dict[path[-1]].append(path)
        for path in right_path:
            if path[-1] in left_dict:
                for l_p in left_dict[path[-1]]:
                    res.add("{};{}".format(l_p[0], path[0]))
        return [list(map(int, p_str.split(";"))) for p_str in res]

    """
    For every rule or relation r_idx, 
    get its detailed information: correct ht, wrong ht, no_idea ht, P, R, F1
    but only record relation_idx, rule_key, P,R,F1
    
    Parameters:
    -----------
    r_idx: int
    index of target relation(r_idx)
    r_path_list: list
    the rules list of relation(r_idx)
    
    Returns:
    -----------
    out: list, [Rule(),Rule(),...]
    A list of rules enchanced
    """

    def enhance_rule(self, r_idx, r_path_list):
        rule_set = set()
        print("Start Enhancing Rule for Relatin: {}".format(self.idx2r[r_idx]))
        rule_list = []
        for idx, r_path in enumerate(r_path_list):
            print("{}/{} Start enchancing Rule: {}".format(idx + 1, len(r_path_list),
                                                           "=>".join(self.display_r_path([r_path])[0])))
            rule = Rule(r_idx, r_path=r_path, rule_key=None)
            if rule.rule_key in rule_set:
                continue
            rule_set.add(rule.rule_key)
            succ = rule.restoreFromMysql()
            if not succ:
                print("Start Fetching passed ht")
                rule.passHT = self.get_passed_ht(r_path)
                assert len(rule.passHT) != 0, "Get Wrong Passed HT"
                print("Start calculating P,R and F1")
                rule.get_P_R_F1(self.node_dict, self.r2ht)
                if rule.P >= 0.001:  # the precision of rule must high enough
                    rule_list.append(rule)
                    while True:
                        if rule.persist2mysql(): break
                    print("Success persisting to mysql.")
            else:
                rule_list.append(rule)
                print("Success loading from MySQL")
        print("Finish Enhancing Rule for Relatin: {}".format(self.idx2r[r_idx]))
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
        print('Start getting model for Relation: {}, Max step: {}'.format(self.idx2r[r_idx], max_step))
        statistics_file = folder + "statistics.txt"
        model_file_path = folder + "model.tar"
        if os.path.exists(statistics_file) and os.path.exists(model_file_path):
            print("Load Model for Relation: {} Max Steps: {}".format(r_idx, max_step))
            with open(statistics_file, 'r', encoding="UTF-8") as f:
                input_size = int(f.readline().strip().split()[1])
            lg = LogisticRegression(input_size)
            lg.loadModel(model_file_path)
            print("Finish loading model from file.")
            return lg

        print("Collect rules for relation: {}".format(self.idx2r[r_idx]))
        rule_list = self.load_rule_from_mysql(r_idx, max_step)
        if len(rule_list) == 0:
            r_path, e_r_path = self.search_path(r_idx, max_step)
            rule_list = self.enhance_rule(r_idx, r_path)

        rule_list.sort(key=lambda one_rule: one_rule.P, reverse=True)
        if len(rule_list) <= top_rules_num:
            top_rules_num = len(rule_list)
        print("The num of Rules collected is {}.".format(top_rules_num))

        with open(statistics_file, 'w', encoding="UTF-8") as f:
            f.write("input_size\t{}\n".format(top_rules_num))

        print("Start collecting Positive/Negetive instances.")
        posi_list = []
        nege_list = []
        for rule in rule_list[:top_rules_num]:
            posi, nege = rule.sample_train_data(posi_num=100, nege_num=100)
            posi_list.extend(posi)
            nege_list.extend(nege)

        train_x = []
        for posi_ht in posi_list:
            feature = self. get_features(rule_list[:top_rules_num], posi_ht)
            train_x.append(feature)
        train_y = list(np.ones(len(posi_list)))

        for nege_ht in nege_list:
            feature = self.get_features(rule_list[:top_rules_num], nege_ht)
            train_x.append(feature)
        train_y.extend(list(np.zeros(len(nege_list))))

        print("Start training model for r:{} max_steps:{} top_rules_num:{}".format(self.idx2r[r_idx], max_step,
                                                                                   top_rules_num))

        lg = LogisticRegression(top_rules_num)
        lg.train(train_x, train_y, epoch=200, mini_batch=500)
        lg.saveModel(model_file_path)
        print('Finish getting model for Relation: {}, Max step: {}'.format(self.idx2r[r_idx], max_step))
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
            if self.is_passed(ht, rule.r_path):
                feature.append(1)
            else:
                feature.append(0)
        return feature

    """
    Get top-K rules for a relation(r_idx) by criterion(P/R/F1)
    Parameters:
    -----------
    r_idx: int 
    index for target relation
    top_k: int 
    num of top rules we want to get
    criterion: str
    'P', 'R' or 'F1', it is the criterion used to sort rule list
    
    Returns:
    -----------
    out: list
    top_k rules
    """

    def get_top_k_rules(self, r_idx, top_k, criterion):
        rule_list = []
        query = "select relation_idx, rule_key from " + database+ \
                " where relation_idx={} order by {} desc".format(r_idx, criterion)
        mycursor = mydb.cursor()
        mycursor.execute(query)
        fetched = mycursor.fetchall()
        for idx, row in enumerate(fetched):
            if idx > top_k - 1: break
            one_rule = Rule(r_idx, None, row[2])
            one_rule.restoreFromMysql()
            rule_list.append(one_rule)
        return rule_list

    '''
    Check if a h t can pass the rule
    Parameters:
    -----------
    ht: list, [head,tail]
    rule: list, [r_idx,r_idx,...]
    
    Returns:
    -----------
    out: boolean
    If ht passed rule, return Ture, False otherwises.
    '''

    def is_passed(self, ht, rule):
        left_node = [ht[0]]
        right_node = [ht[-1]]
        left_step = 0
        right_step = 0
        while len(rule) - (left_step + right_step) > 0:
            temp_left = []
            temp_right = []
            if len(left_node) < len(right_node):
                left_step += 1
                r_idx = rule[left_step - 1]
                for e_idx in left_node:
                    c_node = self.node_dict[e_idx]
                    for tail in c_node.get_tails(r_idx):
                        temp_left.append(tail)
                left_node = temp_left
            else:
                right_step += 1
                r_idx = rule[-right_step]
                inv_r_idx = self.convert_r(r_idx)
                for e_idx in right_node:
                    c_node = self.node_dict[e_idx]
                    for tail in c_node.get_tails(inv_r_idx):
                        temp_right.append(tail)
                right_node = temp_right
        left_set = set()
        for e_idx in left_node:
            left_set.add(e_idx)
        for e_idx in right_node:
            if e_idx in left_set:
                return True
        return False


if __name__ == "__main__":
    util = Util()
    source_folder = "./source/"
    source_folder = "F:\\Data\\FB15K-237\\source\\"
    output_folder = "F:\\Data\\FB15K-237\\output\\"
    e2idx_file = source_folder + "e2idx.txt"
    r2idx_file = source_folder + "r2idx.txt"
    triple2idx_file = source_folder + "triple2idx.txt"

    # relation_list = ['<http://dbpedia.org/ontology/director>',
    #                  '<http://dbpedia.org/ontology/starring>',
    #                  '<http://dbpedia.org/ontology/birthPlace>']

    relation_list = ['/film/film/language']
    graph = Graph(e2idx_file, r2idx_file, triple2idx_file)
    graph.load_data()
    r_idx_list = [graph.r2idx[relation] for relation in relation_list]
    r_rules_dict = {}
    for idx, r_idx in enumerate(r_idx_list):
        folder = output_folder + util.gen_prefix(relation_list[idx]) + file_path_seg
        if not os.path.isdir(folder):
            os.makedirs(folder)
        graph.get_r_model(r_idx=r_idx, max_step=3, top_rules_num=200, folder=folder)
        r_rules_dict[r_idx] = graph.get_top_k_rules(r_idx, 5, 'P')

    

    for key in r_rules_dict.keys():
        print("Relation: {}".format(graph.idx2r[key]))
        for p in graph.display_r_path(r_rules_dict[key]):
            print("=>".join(p))
        print("\n")

    # displayed_path = graph.display_r_path(r_path_list)
    # displayed_path = graph.display_e_r_path(e_r_path_list)
    # for p in displayed_path:
    #     print("=>".join(p))
