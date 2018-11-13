from RuleBased.BiSearch.MyThread import MyThread
from RuleBased.BiSearch.Triple import Node, Rule
import random
import numpy as np
import os

from RuleBased.Classifier import LogisticRegression
from RuleBased.Params import rule_seg, mydb, file_path_seg, database, ht_conn, ht_seg, sampled_num_to_search_rule, \
    top_frequency_rule_num, epoch, mini_batch, rule_num4train, max_step
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
    Connect two path found by unidirection search
    Parameters:
    -----------
    left_path: list [[-1,e_idx,r_idx,...],[]]
    right_path: list [[-1,e_idx,r_idx,...],[],...]
    
    Returns:
    -----------
    out: list [[-1,e_idx,r_idx,...,e_idx,-1],[],[],...]
    '''

    def join_e_r_path(self, left_path, right_path):
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
            temp_res = self.join_e_r_path(left_path[left_len], right_path[right_len])
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
    out: list, [Rule(),Rule(),...]
    the list of rules for relation(r_idx) loaded from mysql
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
                    for tail in c_node.get_tails_of_r_idx(r_idx):
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
                    for tail in c_node.get_tails_of_r_idx(inv_r_idx):
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

    '''
    Fetch positive/negetive instances of rules.
    Parameters:
    -----------
    rule_list: list, [Rule(),Rule(),...]
    A list of Rule object.
    
    Returns:
    -----------
    posi_list: list, [[h,t],[h,t],...]
    nege_list: list, [[h,t],[h,t],...]
    '''

    def fetch_posi_nege_of_rules(self, rule_list):
        print("Start collecting Positive/Negetive instances.")
        posi_list = []
        nege_list = []
        for rule in rule_list:
            posi, nege = rule.sample_train_data(posi_num=100, nege_num=100)
            rule_display = "=>".join(self.display_r_path([rule.r_path])[0])
            print("Rule: {}, Posi_Num: {}, Nege_Num: {}.".format(rule_display, len(posi), len(nege)))
            posi_list.extend(posi)
            nege_list.extend(nege)
        return posi_list, nege_list

    '''
    Collect rules for relation(r_idx)
    Parameters:
    -----------
    statistics_file: string
    It is a file path, this file records the num of rules to train.
    r_idx: int
    index of target relation.
    
    Return:
    -----------
    out: list, [Rule(),Rule(),...]
    A list of rule obejct, sorted by precision.   
    '''

    def collect_rules_for_r_idx(self, r_idx, statistics_file):
        print("Collect rules for relation: {}".format(self.idx2r[r_idx]))
        rule_list = self.load_rule_from_mysql(r_idx, max_step)
        if len(rule_list) == 0:
            r_path, e_r_path = self.search_path(r_idx, max_step)
            rule_list = self.enhance_rule(r_idx, r_path)
            print("Get rules by search.")
        else:
            print("Load rules from mysql.")

        rule_list.sort(key=lambda one_rule: one_rule.P, reverse=True)

        rules_to_use = rule_num4train
        if len(rule_list) < rule_num4train:
            rules_to_use = len(rule_list)
        print("The num of Rules collected is {}.".format(rules_to_use))

        with open(statistics_file, 'w', encoding="UTF-8") as f:
            f.write("input_size\t{}\n".format(rules_to_use))
        print("Writing input_size to file.")

        return rule_list[:rule_num4train]

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

    def get_r_model(self, r_idx, folder):
        print('Get model for Relation: {}'.format(self.idx2r[r_idx]))
        statistics_file = folder + "statistics.txt"
        train_x_file = folder + "train_x.npy"
        train_y_file = folder + "train_y.npy"
        model_file_path = folder + "model.tar"
        if os.path.exists(statistics_file) and os.path.exists(model_file_path):
            print("Load Model for Relation: {}".format(r_idx))
            with open(statistics_file, 'r', encoding="UTF-8") as f:
                input_size = int(f.readline().strip().split()[1])
            lg = LogisticRegression(input_size)
            lg.loadModel(model_file_path)
            print("Finish loading model from file.")
            return lg

        rule_list = self.collect_rules_for_r_idx(r_idx, statistics_file)

        print("Train model for r:{}, rules_used_num:{}".format(self.idx2r[r_idx], len(rule_list)))

        posi_list, nege_list = self.fetch_posi_nege_of_rules(rule_list)

        print("Start getting features for positives.")
        train_x = self.get_features(rule_list, posi_list)
        train_y = list(np.ones(len(posi_list)))

        print("Start getting features for negetives.")
        train_x.extend(self.get_features(rule_list, nege_list))
        train_y.extend(list(np.zeros(len(nege_list))))

        print("Save features to file.")
        np.save(train_x_file, np.array(train_x))
        np.save(train_y_file, np.array(train_y))

        lg = LogisticRegression(len(rule_list))

        zipped_x_y = list(zip(train_x, train_y))
        np.random.shuffle(zipped_x_y)
        train_x, train_y = zip(*zipped_x_y)
        lg.train(train_x, train_y, epoch=epoch, mini_batch=mini_batch)
        lg.saveModel(model_file_path)
        print('Finish getting model for Relation: {}, Max step: {}'.format(self.idx2r[r_idx], max_step))
        return lg

    '''
    Test ability of Logistic Regression
    Parameters:
    -----------
    r_name: string
    name of target relatoin
    folder: string
    target folder under which data was saved.
    '''

    def test_model(self, r_name, folder):
        print("Test model of r:{}".format(r_name))
        statistics_file = folder + "statistics.txt"
        train_x_file = folder + "train_x.npy"
        train_y_file = folder + "train_y.npy"

        train_x = np.load(train_x_file)
        train_y = np.load(train_y_file)

        with open(statistics_file, 'r', encoding="UTF-8") as f:
            input_size = int(f.readline().strip().split()[1])

        train_num = int(len(train_x) * 0.8)
        zipped_x_y = list(zip(train_x, train_y))

        for i in range(10):
            lg = LogisticRegression(input_size)
            np.random.shuffle(zipped_x_y)
            temp_x, temp_y = zip(*zipped_x_y)
            temp_train_x = temp_x[:train_num]
            temp_train_y = temp_y[:train_num]
            temp_test_x = temp_x[train_num + 1:]
            temp_test_y = temp_y[train_num + 1:]
            lg.train(temp_train_x, temp_train_y, epoch=epoch, mini_batch=mini_batch)
            precision = lg.test(temp_test_x, temp_test_y)
            print("Test round {}/{}, Prec: {}".format(i + 1, 10, precision))

    """
    Get features for one pair of h and t
    Parameters:
    -----------
    rule_list: list, [[r_idx,r_idx,...],[],[],[],...]
    It stores a list of rules.
    ht: list, [[h,t],[h,t],[h,t],..]
    a list of h and t
    
    Returns:
    -----------
    out: list
    A list of features, every entry represents if a rule is passed.
    """

    def get_features(self, rule_list, ht_list):
        train_x = []
        for ht in ht_list:
            feature = []
            for rule in rule_list:
                feature.append(int(self.is_passed(ht, rule.r_path)))
            train_x.append(feature)
        return train_x

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
    out: list, [Rule(),Rule(),Rule(),...]
    top_k rules
    """

    def get_top_k_rules(self, r_idx, top_k, criterion):
        rule_list = []
        query = "select relation_idx, rule_key from " + database + \
                " where relation_idx={} order by {} desc".format(r_idx, criterion)
        mycursor = mydb.cursor()
        mycursor.execute(query)
        fetched = mycursor.fetchall()
        for idx, row in enumerate(fetched):
            if idx > top_k - 1: break
            one_rule = Rule(r_idx, None, row[1])
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
                    for tail in c_node.get_tails_of_r_idx(r_idx):
                        temp_left.append(tail)
                left_node = temp_left
            else:
                right_step += 1
                r_idx = rule[-right_step]
                inv_r_idx = self.convert_r(r_idx)
                for e_idx in right_node:
                    c_node = self.node_dict[e_idx]
                    for tail in c_node.get_tails_of_r_idx(inv_r_idx):
                        temp_right.append(tail)
                right_node = temp_right
        left_set = set()
        for e_idx in left_node:
            left_set.add(e_idx)
        for e_idx in right_node:
            if e_idx in left_set:
                return True
        return False

    '''
    Find h and t which can pass at least one rule of rule_list
    Parameters:
    -----------
    h_idx_list: list, a list of h_idx, [e_idx,e_idx,...]
    t_idx_list: list, a list of t_idx, [e_idx,e_idx,...]
    rule_list: list, a list of rules, [[r_idx,r_idx],[r_idx,r_idx,r_idx],...]
    
    Returns:
    -----------
    out1: list, [[h_idx,t_idx],[h_idx,t_idx],...]
    A list of [h_idx,t_idx] that can pass the rule list.
    There is no duplicate in out list.
    out2: list, the token of out1, token => h,t
    
    '''

    def pass_verify(self, h_idx_list, t_idx_list, rule_list):
        # h_idx_list = [self.e2idx[h_name] for h_name in h_name_list]
        # t_idx_list = [self.e2idx[t_name] for t_name in t_name_list]
        ht_is_passed = {}
        for rule in rule_list:
            thread_list = []
            t_idx2ht_token = {}
            for h_idx in h_idx_list:
                for t_idx in t_idx_list:
                    ht_token = "{}{}{}".format(h_idx, ht_conn, t_idx)
                    if ht_token in ht_is_passed: continue
                    th = MyThread(self.is_passed, ([h_idx, t_idx], rule))
                    thread_list.append(th)
                    t_idx2ht_token[len(thread_list) - 1] = ht_token

            [th.start for th in thread_list]
            [th.join() for th in thread_list]

            for th_idx in range(len(thread_list)):
                ht_token = t_idx2ht_token[th_idx]
                if thread_list[th_idx].get_result():
                    ht_is_passed[ht_token] = 1

        return [ht_token.split(ht_conn) for ht_token in ht_is_passed.keys()], set(ht_is_passed.keys())

    '''
    Extend ht_path after relation(r_idx), from left to right
    Parameters:
    -----------
    ht_path: list, [[h,m1,m2,..,t],[],[],...]
    r_idx: int, index of relation
    
    Returns:
    -----------
    out: list, [[h,m1,m2,..,t,new_t],[],[],...]
    '''

    def get_ht_path_from_left(self, ht_path, r_idx):
        res_ht_path = []
        for ht in ht_path:
            c_node = self.node_dict[ht[-1]]
            for tail in c_node.get_tails_of_r_idx(r_idx=r_idx):
                temp = ht.copy()
                temp.append(tail)
                res_ht_path.append(temp)
        return res_ht_path

    '''
    Extend ht_path after relation(r_idx), from right to left
    Parameters:
    -----------
    ht_path: list, [[h,m1,m2,..,t],[],[],...]
    r_idx: int, index of relation

    Returns:
    -----------
    out: list, [[h,m1,m2,..,t,new_t],[],[],...]
    '''

    def get_ht_path_from_right(self, ht_path, r_idx):
        res_ht_path = []
        inv_r_idx = self.convert_r(r_idx)
        for ht in ht_path:
            c_node = self.node_dict[ht[-1]]
            for tail in c_node.get_tails_of_r_idx(r_idx=inv_r_idx):
                temp = ht.copy()
                temp.append(tail)
                res_ht_path.append(temp)
        return res_ht_path

    '''
    Get ht that can pass the rule, one of h_idx_list and t_idx_list is empty.
    Parameters:
    -----------
    h_idx_list: list, list of h_idx, [h_idx,h_idx,...]
    t_idx_list: list, list of t_idx, [t_idx,t_idx,...]
    rule: list, list of r_idx, [r_idx,r_idx,r_idx,...]
    passed_ht_token_set: set, (['h,t','h,t',...],set())
    a set of tokens of passed ht
    
    Returns:
    -----------
    out1: list, [[h_idx,t_idx],[h_idx,t_idx],[h_idx,t_idx],...]
    '''

    def get_ht_from_one_end(self, h_idx_list, t_idx_list, rule, passed_ht_token_set):
        assert (len(h_idx_list) != 0 or len(t_idx_list) != 0), "Can't get ht from one end."

        def ht2token(h, t):
            return ht_conn.join([str(h), str(t)])

        res_left_path_list = [[h_idx] for h_idx in h_idx_list]
        res_right_path_list = [[t_idx] for t_idx in t_idx_list]
        left_empty = True if len(h_idx_list) == 0 else False

        if not left_empty:
            for r_idx in rule:
                res_left_path_list = self.get_ht_path_from_left(res_left_path_list, r_idx)
        else:
            for r_idx in list(reversed(rule)):
                res_right_path_list = self.get_ht_path_from_right(res_right_path_list, r_idx)

        temp_res_ht_list = res_right_path_list if left_empty else res_left_path_list
        res_ht_list = []

        for ht_path in temp_res_ht_list:
            h = ht_path[-1] if left_empty else ht_path[0]
            t = ht_path[0] if left_empty else ht_path[-1]
            token = ht2token(h, t)
            if token not in passed_ht_token_set:
                passed_ht_token_set.add(token)
                res_ht_list.append([h, t])
        # left_step = 0
        # right_step = 0
        # while len(rule) - (left_step + right_step) > 0:
        #     if left_step == 0 and right_step == 0:
        #         from_left = len(res_left_path_list) > len(res_right_path_list)
        #     else:
        #         from_left = len(res_left_path_list) < len(res_right_path_list)
        #     if from_left:
        #         left_step += 1
        #         res_left_path_list = self.get_ht_path_from_left(res_left_path_list, rule[left_step - 1])
        #     else:
        #         right_step += 1
        #         res_right_path_list = self.get_ht_path_from_right(res_right_path_list, rule[-right_step])
        #
        # res_token_set = set()
        # res_ht_list = []
        # if left_empty and left_step == 0:
        #     for right_path in res_right_path_list:
        #         token = ht2token(right_path[-1], right_path[0])
        #         if token not in res_token_set:
        #             res_token_set.add(token)
        #             res_ht_list.append([right_path[-1], right_path[0]])
        # elif not left_empty and right_step == 0:
        #     for left_path in res_left_path_list:
        #         token = ht2token(left_path[0], left_path[-1])
        #         if token not in res_token_set:
        #             res_token_set.add(token)
        #             res_ht_list.append([left_path[0], left_path[-1]])
        # else:
        #     left_end_dict = {}
        #     for left_path in res_left_path_list:
        #         if left_path[-1] not in left_end_dict:
        #             left_end_dict[left_path[-1]] = []
        #         left_end_dict[left_path[-1]].append(left_path)
        #
        #     for right_path in res_right_path_list:
        #         end_point = right_path[-1]
        #         if end_point in left_end_dict:
        #             for left_path in left_end_dict[end_point]:
        #                 res_token_set.add(ht2token(left_path[0], right_path[0]))
        #                 res_ht_list.append([left_path[0], right_path[0]])
        return res_ht_list


if __name__ == "__main__":
    util = Util()
    # source_folder = "./source/"
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
        graph.get_r_model(r_idx=r_idx, folder=folder)
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
