from ALogger import ALogger
from RuleBased.Triple import Node, Rule
import random
import numpy as np
import os
import queue
import time
import threading

from RuleBased.Classifier import LogisticRegression
from RuleBased.Params import rule_seg, ht_conn, ht_seg, epoch_num, \
    rule_num4train, max_step, filter_inv_pattern, \
    check_time_for_get_passed_ht, time_limit_for_get_passed_ht, branch_node_limit, limit_branch_node_num, restrain_num, \
    restrain_num_of_posis_neges, train_data_num_upper_limit, alpha, pca_or_cwa, batch_size
from Util import Util
from RuleBased.EmbedGraph import EGraph


class MyThread(threading.Thread):
    def __init__(self, func, args):
        threading.Thread.__init__(self)
        self.func = func
        self.args = args
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None


class Graph:
    def __init__(self):
        self.logger = ALogger("Graph", True).getLogger()
        self.util = Util()

        self._build_container()

    def _build_train_file_path(self, args):

        self.root_folder = args.root_folder
        self.e2idx_file = args.e2id_file
        self.r2idx_file = args.r2id_file
        self.triple2idx_file = args.triple2id_file

        self.model_folder = args.model_folder
        self.rule_file = args.rule_file
        self.rule_num_to_use_file = args.rule_num_to_use_file
        self.train_id_data_file = args.train_id_data_file
        self.train_feature_data_file = args.train_feature_data_file
        self.model_file = args.model_file

    def _build_search_file_path(self, args):
        self.e2idx_file = args.e2id_file
        self.r2idx_file = args.r2id_file
        self.triple2idx_file = args.triple2id_file

        self.qe_res_all = args.qe_res_all
        self.qe_res_topk = args.qe_res_topk

    def _build_container(self):
        self.e_name2id = {}
        self.e_id2name = {}
        self.r_name2id = {}
        self.r_id2name = {}
        self.node_dict = {}

        """
        {
            r_idx:[[h,t],...,[h,t]],
            r_idx:[[h,t],...,[h,t]],
            ...
        }
        """
        self.r2ht = {}

        self.ekg = None

    def load_er_embedding(self):
        if self.ekg is None:
            self.logger.info("Load embedding")
            self.ekg = EGraph()

    def get_er_embedding(self):
        self.load_er_embedding()
        return self.ekg.e2vec, self.ekg.r2vec

    def load_data(self):
        if len(list(self.r_name2id.keys())) != 0:
            return
        self.logger.info("Load Graph: {}.".format(self.e2idx_file.split('/')[-2]))
        with open(self.e2idx_file, 'r', encoding="UTF-8") as f:
            for line in f.readlines()[1:]:
                name, idx = line.strip().split()
                idx = int(idx)
                self.e_name2id[name] = idx
                self.e_id2name[idx] = name
        with open(self.r2idx_file, 'r', encoding="UTF-8") as f:
            for line in f.readlines()[1:]:
                name, idx = line.strip().split()
                idx = int(idx)
                self.r_name2id[name] = idx
                self.r_id2name[idx] = name
                self.r2ht[idx] = []
        with open(self.triple2idx_file, 'r', encoding='UTF-8') as f:
            for line in f.readlines()[1:]:
                h, t, r = line.strip().split()
                h = int(h)
                r = int(r)
                t = int(t)
                self.r2ht[r].append([h, t])
                if h not in self.node_dict:
                    self.node_dict[h] = Node(h)
                if t not in self.node_dict:
                    self.node_dict[t] = Node(t)
                self.node_dict[h].addPath(r=r, e=t)
                inv_r = "inv_{}".format(self.r_id2name[r])
                if inv_r in self.r_name2id:
                    self.r2ht[self.r_name2id[inv_r]].append([t, h])
                    self.node_dict[t].addPath(r=self.r_name2id[inv_r], e=h)

    '''
    If a fact(h_idx,r_idx,t_idx) is in this graph.
    Parameters:
    -----------
    h_idx: int, the index of head
    r_idx: int, the index of relation
    t_idx: int, the index of tail
    
    Returns:
    ----------
    out: boolean, true if (h_idx,r_idx,t_idx) in graph, false else.
    '''

    def has_fact(self, h_idx, r_idx, t_idx):
        temp_node = self.node_dict[h_idx]
        has_r, has_t = temp_node.has_r_t(r_idx, t_idx)
        return has_r and has_t

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
                                name = self.r_id2name[r_p[r_p_i]]
                                if name.startswith("inv_"):
                                    name = name[4:]
                                else:
                                    name = "inv_" + name
                                temp_l.append(self.r_name2id[name])
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
    for example, when step = 3, we get path whose length is 1, 2 and 3
    
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
            temp_res = self.join_e_r_path(left_path[left_len],
                                          right_path[right_len])
            res.extend(temp_res)
        return res

    '''
    Check if two relation is inverse to each other.
    Parameters:
    -----------
    r_idx1: int
    index of relation1
    r_idx2: int
    index of relation2
    
    Returns:
    -----------
    out: boolean
    True if r_idx1 and r_idx2 is inverse to each other, false otherwise.
    '''

    def is_inverse_r_idx(self, r_idx1, r_idx2):
        r_name1 = self.r_id2name[r_idx1]
        r_name2 = self.r_id2name[r_idx2]
        if r_name1.startswith("inv_") and r_name1 == "inv_{}".format(r_name2):
            return True
        if r_name2.startswith("inv_") and r_name2 == "inv_{}".format(r_name1):
            return True
        return False

    '''
    Check if a r_path has inverse pattern relation
    Parameters:
    -----------
    r_path: list, [r_idx,r_idx,...]
    a list of relation path
    
    Returns:
    -----------
    out: boolean
    True if r_path has inverse pattern relation, false otherwish.
    '''

    def has_inverse_r_in_r_path(self, r_path):
        for r_path_i in range(len(r_path) - 1):
            if self.is_inverse_r_idx(r_path[r_path_i], r_path[r_path_i + 1]):
                return True
        return False

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
                    if i >= 1 and path.e == n[-3]:
                        continue
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
                    temp_display.append(self.e_id2name[path[idx]])
                else:
                    temp_display.append(self.r_id2name[path[idx]])
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
            temp = [self.r_id2name[r_idx] for r_idx in r_path]
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
    first list is the path of r, [[r_idx,r_idx,r_idx,...],...,[r_idx,r_idx,r_idx,...]]
    second list is the path of e and r, [[-1,e_idx,r_idx,e_idx,r_iex,...,-1],...,[-1,e_idx,r_idx,e_idx,r_iex,...,-1]]
    """

    def search_path(self, r_id, max_step):
        search_r_path_num = {}
        searched_r_path = {}
        searched_e_r_path = []

        train_data = self.get_train_data_4_r_idx(r_id, self.train_id_data_file)
        ht_array = train_data[np.where(train_data[:, 2] == 1)]
        ht_array = ht_array[:, 0:2]

        self.logger.info("Search Path for {}/{}, Train Num:{}".format(r_id, self.r_id2name[r_id], ht_array.shape[0]))
        for idx, ht in enumerate(ht_array):
            ht_r_path_key_set = set()
            h = ht[0]
            t = ht[1]
            if (idx + 1) % 100 == 0:
                self.logger.info(
                    '{}/{} R: {} H: {} T: {}'.format(idx + 1, ht_array.shape[0], self.r_id2name[r_id],
                                                     self.e_id2name[h],
                                                     self.e_id2name[t]))
            path_found = self.search_bidirect(h, t, max_step)
            for p in path_found:
                r_path = self.extract_r_path(p)
                if filter_inv_pattern and self.has_inverse_r_in_r_path(r_path):
                    continue
                r_path_key = rule_seg.join(map(str, r_path))
                if len(r_path) == 1 and r_path[0] == r_id:
                    continue
                searched_e_r_path.append(p)

                if r_path_key not in search_r_path_num:
                    searched_r_path[r_path_key] = r_path
                    search_r_path_num[r_path_key] = 1
                elif r_path_key not in ht_r_path_key_set:
                    search_r_path_num[r_path_key] += 1
                    ht_r_path_key_set.add(r_path_key)

        res_r_path_list = []
        for r_path_key in search_r_path_num:
            ratio = 1.0 * search_r_path_num[r_path_key] / ht_array.shape[0]
            if ratio >= alpha:
                res_r_path_list.append(searched_r_path[r_path_key])
        # for key, value in list(
        #         sorted(
        #             search_r_path_num.items(),
        #             key=lambda d: d[1],
        #             reverse=True))[:top_frequency_rule_num]:
        #     res_r_path_list.append(searched_r_path[key])
        return res_r_path_list, searched_e_r_path

    '''
    Load rules for relation(r_idx)
    Parameters:
    -----------
    r_name: str, dbo:birthPlace

    Returns:
    -----------
    out: list, [Rule(),Rule(),...]
    A rule object has rule_id_key, p, r, f
    '''

    def load_rule_obj_from_file(self, r_name):
        self.logger.info("Rule_file: {}".format(self.rule_file))
        if not os.path.exists(self.rule_file):
            return []
        rule_obj_list = []
        for line in open(self.rule_file, 'r', encoding="UTF-8").readlines():
            _, name_key, p, r, f = line.split("\t")
            r_name_rule = name_key.split(rule_seg)
            r_id_rule = [self.r_name2id[tr_name] for tr_name in r_name_rule]

            tmp_rule_obj = Rule(self.r_name2id[r_name])
            tmp_rule_obj.set_r_id_rule(r_id_rule)
            tmp_rule_obj.set_r_name_rule(r_name_rule)
            tmp_rule_obj.set_P_R_F1(p, r, f)

            rule_obj_list.append(tmp_rule_obj)
        return rule_obj_list

    # '''
    # Load rules for relation(r_idx), the length of rules is under max_step
    # Parameters:
    # -----------
    # r_idx: int, index of relation
    # max_step: int, max length of rule of relation(r_idx)
    #
    # Returns:
    # -----------
    # out: list, [Rule(),Rule(),...]
    # the list of rules for relation(r_idx) loaded from mysql
    # '''
    #
    # def load_rule_from_mysql(self, r_idx, max_step):
    #     rule_list = []
    #     sql = "select * from " + database + " where relation_idx = {} and rule_len <= {};".format(
    #         r_idx, max_step)
    #     mycursor = mydb.cursor()
    #     mycursor.execute(sql)
    #     fetched = mycursor.fetchall()
    #     for row in fetched:
    #         rule = Rule(r_idx, None, row[2])
    #         rule.rule_len = int(row[3])
    #         if len(row[4]) != 0:
    #             rule.correct_ht = [
    #                 list(map(int, ht2)) for ht2 in
    #                 [ht.split(ht_conn) for ht in row[4].split(ht_seg)]
    #             ]
    #         if len(row[5]) != 0:
    #             rule.wrong_ht = [
    #                 list(map(int, ht2)) for ht2 in
    #                 [ht.split(ht_conn) for ht in row[5].split(ht_seg)]
    #             ]
    #         if len(row[6]) != 0:
    #             rule.no_idea_ht = [
    #                 list(map(int, ht2)) for ht2 in
    #                 [ht.split(ht_conn) for ht in row[6].split(ht_seg)]
    #             ]
    #         rule.P = row[7]
    #         rule.R = row[8]
    #         rule.F1 = row[9]
    #         rule_list.append(rule)
    #     return rule_list

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
        name = self.r_id2name[r_idx]
        if name.startswith("inv_"):
            name = name[4:]
        else:
            name = "inv_" + name
        return self.r_name2id[name]

    """
    Get a list of ht which can pass r_path
    Parameters:
    -----------
    r_path: list 
    a rule: [r_idx,r_idx,r_idx,...], for example: [2,3,6,...]
    
    Return:
    -----------
    out1: boolean
    True if time exceeds, False otherwish.
    out2: list, [[h,t],[h,t],[h,t],...]
    list of passed ht
    """

    def get_passed_ht(self, r_path):
        start_time = time.time()

        def time_exceed():
            end_time = time.time()
            if check_time_for_get_passed_ht and (
                    end_time - start_time) > time_limit_for_get_passed_ht:
                print("Elapsed: {}".format(end_time - start_time))
                return True

        if len(r_path) == 1:
            return False, self.r2ht[r_path[0]]
        left_path = self.r2ht[r_path[0]]
        left_step = 1
        inv_r_idx = self.convert_r(r_path[-1])
        right_path = self.r2ht[inv_r_idx]
        right_step = 1
        while len(r_path) - (left_step + right_step) > 0:
            print("Left Path length: {}.".format(len(left_path)))
            print("Right Path length: {}.".format(len(right_path)))
            if limit_branch_node_num and len(left_path) > branch_node_limit:
                left_path = random.sample(left_path, branch_node_limit)
            if limit_branch_node_num and len(right_path) > branch_node_limit:
                right_path = random.sample(right_path, branch_node_limit)

            temp_left_path = []
            temp_right_path = []
            if len(left_path) < len(right_path):
                left_step += 1
                r_idx = r_path[left_step - 1]
                for ht in left_path:
                    c_node = self.node_dict[ht[-1]]
                    for tail in c_node.get_tails_of_r_idx(r_idx):
                        if time_exceed():
                            return True, []
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
                        if time_exceed():
                            return True, []
                        temp_ht = ht.copy()
                        temp_ht.append(tail)
                        temp_right_path.append(temp_ht)
                right_path = temp_right_path
        res = {}
        left_dict = {}
        for path in left_path:
            if path[-1] not in left_dict:
                if time_exceed():
                    return True, []
                left_dict[path[-1]] = []
            left_dict[path[-1]].append(path)
        for path in right_path:
            if path[-1] in left_dict:
                for l_p in left_dict[path[-1]]:
                    if time_exceed():
                        return True, []
                    temp_token = "{};{}".format(l_p[0], path[0])
                    if temp_token not in res:
                        res[temp_token] = [l_p[0], path[0]]
        print("Elapsed: {}".format(time.time() - start_time))
        return False, [res[key] for key in res]

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

    def rule_prec_reca_f(self, r_idx, r_path_list):
        rule_set = set()
        rule_list = []
        for idx, r_id_rule in enumerate(r_path_list):
            r_name_rule = [self.r_id2name[r_idx] for r_idx in r_id_rule]

            # self.logger.info("{}/{}, R:{}, Rule: {}".format(
            #     idx + 1, len(r_path_list), self.r_id2name[r_idx], rule_seg.join(r_name_rule)))

            rule = Rule(r_idx)
            rule.set_r_id_rule(r_id_rule)
            rule.set_r_name_rule(r_name_rule)

            if rule.rule_id_key in rule_set:
                continue

            rule_set.add(rule.rule_id_key)
            time_exceeds, rule.passHT = self.get_passed_ht(r_id_rule)
            if time_exceeds:
                self.logger.info("Fetching passed ht time exceeds. Abandon this rule.")
                continue
            assert len(rule.passHT) != 0, "Get Wrong Passed HT"
            rule.get_P_R_F1(self.node_dict, self.r2ht)
            rule_list.append(rule)
            # if rule.P >= 0.001:  # the precision of rule must high enough
            #     rule_list.append(rule)
            #     # while True:
            #     #     if rule.persist2mysql():
            #     #         break
            #     # print("Success persisting to mysql.")
            # else:
            #     self.logger.info("Prec: {}, low, abandon.".format(rule.P))
            rule_list.sort(key=lambda x: x.F1, reverse=True)
        return rule_list[:1000]

    # '''
    # Fetch positive/negetive instances of rules.
    # Parameters:
    # -----------
    # rule_list: list, [Rule(),Rule(),...]
    # A list of Rule object.
    #
    # Returns:
    # -----------
    # posi_list: list, [[h,t],[h,t],...]
    # nege_list: list, [[h,t],[h,t],...]
    # '''
    #
    # def fetch_posi_nege_of_rules(self, rule_list):
    #     print("Start collecting Positive/Negetive instances.")
    #     posi_list = []
    #     nege_list = []
    #     for rule in rule_list:
    #         posi, nege = rule.sample_train_data(posi_num=100, nege_num=100)
    #         rule_display = "=>".join(self.display_r_path([rule.r_path])[0])
    #         print("Rule: {}, Posi_Num: {}, Nege_Num: {}.".format(
    #             rule_display, len(posi), len(nege)))
    #         posi_list.extend(posi)
    #         nege_list.extend(nege)
    #     return posi_list, nege_list

    '''
    Get train data for r_idx
    _______________________
    Parameters:
    r_id: id of relation
    train_data_file: file to store training data (.npy)
    
    out:
    train aata
    '''

    def get_train_data_4_r_idx(self, r_id, train_data_file):

        self.logger.info("Get training id data for relation: {}/{}".format(r_id, self.r_id2name[r_id]))

        if os.path.exists(train_data_file):
            return np.load(train_data_file)

        ht_list = self.r2ht[r_id]

        if len(ht_list) <= train_data_num_upper_limit:
            positive_ht = ht_list[:]
        else:
            positive_ht = random.sample(ht_list, k=train_data_num_upper_limit)

        positive_ht = np.array(positive_ht)
        token_dict = ["{}{}{}".format(h_id, ht_seg, t_id) for h_id, t_id in list(positive_ht)]

        h_id_array = positive_ht[:, 0]
        t_id_array = positive_ht[:, 1]
        negetive_ht = []
        for h_id in h_id_array:
            cnt = 0
            while True:
                t_id = np.random.choice(t_id_array)
                token = "{}{}{}".format(h_id, ht_seg, t_id)
                if token not in token_dict:
                    negetive_ht.append([h_id, t_id])
                    break
                cnt += 1
                if cnt >= 10:
                    break
        negetive_ht = np.array(negetive_ht)
        label = np.vstack((np.ones([positive_ht.shape[0], 1], dtype=np.int32),
                           np.zeros([negetive_ht.shape[0], 1], dtype=np.int32)))
        train_input = np.vstack((positive_ht, negetive_ht))

        train_id_data = np.hstack((train_input, label))
        np.save(train_data_file, train_id_data)

        self.logger.info("Save training id data to {}".format(train_data_file))
        return train_id_data

    '''
    Collect rules for relation(r_idx)
    Parameters:
    -----------
    r_idx: int
    index of target relation.
    
    Return:
    -----------
    out: list, [Rule(),Rule(),...]
    A list of rule obejct, sorted by precision descendingly.   
    '''

    def collect_rules_for_r_idx(self, r_idx):
        self.logger.info("Collect rules for : {}/{}".format(r_idx, self.r_id2name[r_idx]))

        rule_obj_list = self.load_rule_obj_from_file(self.r_id2name[r_idx])
        if len(rule_obj_list) == 0:
            r_path, e_r_path = self.search_path(r_idx, max_step)
            rule_obj_list = self.rule_prec_reca_f(r_idx, r_path)
            with open(self.rule_file, 'w', encoding="UTF-8") as f:
                for rule_obj in rule_obj_list:
                    f.write("{}\n".format(rule_obj.toStr()))

        # rule_obj_list.sort(key=lambda one_rule: one_rule.P, reverse=True)

        rule_num_2_use = rule_num4train
        if len(rule_obj_list) < rule_num4train:
            rule_num_2_use = len(rule_obj_list)
        self.logger.info("Rule num to use: {}.".format(rule_num_2_use))

        with open(self.rule_num_to_use_file, 'w', encoding="UTF-8") as f:
            f.write("input_size\t{}\n".format(rule_num_2_use))

        return rule_obj_list[:rule_num4train]

    # '''
    # Give rules used for training label.
    # Label is an ascending integer, indicating the index of rule in final feature vector.
    # Parameters:
    # -----------
    # rule_list: list
    # [Rule(),Rule(),Rule(),...]
    # '''
    #
    # def persist_rule4train(self, rule_list):
    #     print("Give training label for rule.")
    #     cnt = 0
    #     for rule_obj in rule_list:
    #         rule_obj.add_train_label(cnt)
    #         cnt += 1
    #     print("Finish giving training label.")

    # '''
    # Get list of rules for train for relation(r_idx)
    # Parameters:
    # -----------
    # r_idx: int
    # idx for relation
    #
    # Returns:
    # -----------
    # out: list, [Rule(),Rule(),...]
    # Rule object only has r_idx, r_path, and rule_key
    # '''
    #
    # def get_rule4train_from_mysql(self, r_idx):
    #     query = "select rule_key from " + database + \
    #             " where relation_idx = {} and rule4train <> {} order by rule4train asc;".format(r_idx, -1)
    #     mycursor = mydb.cursor()
    #     mycursor.execute(query)
    #     fetched = mycursor.fetchall()
    #     assert len(fetched) != 0, "Get rule4train error!"
    #
    #     rule_obj_list = []
    #     cnt = 0
    #     for row in fetched:
    #         if cnt >= rule_num4train:
    #             continue
    #         cnt += 1
    #         rule_key = row[0]
    #         rule_obj_list.append(Rule(r_idx, None, rule_key))
    #     return rule_obj_list

    """
    Train or load a test_model for relation: r_idx
    Parameters
    ----------
    r_idx: the index of a relation
    max_step: max steps for the rules
    top_rules_num: the rule num used to train test_model
    folder: the folder under which thd output test_model is saved
    
    Returns:
    ----------
    out: A trained test_model of relation r_idx
    
    """

    def get_pra_model4r(self, r_idx):
        if os.path.exists(self.rule_num_to_use_file) and os.path.exists(self.model_file):
            with open(self.rule_num_to_use_file, 'r', encoding="UTF-8") as f:
                input_size = int(f.readline().strip().split()[1])
            self.logger.info("Load Model R: {}, feature size: {}.".format(self.r_id2name[r_idx], input_size))
            lg = LogisticRegression(input_size)
            lg.loadModel(self.model_file)
            return lg

        self.load_data()
        if pca_or_cwa == "pca":
            self.load_er_embedding()

        rule_list = self.collect_rules_for_r_idx(r_idx)

        self.logger.info("Train model R:{}, rules_used_num:{}".format(self.r_id2name[r_idx], len(rule_list)))

        train_id_data = np.load(self.train_id_data_file)
        train_feature_data = self.load_train_feature_data(r_idx, rule_list, train_id_data)

        lg = LogisticRegression(len(rule_list))

        self.logger.info("Start training")
        lg.update(train_feature_data, epoch_num, batch_size)
        lg.saveModel(self.model_file)
        return lg

    def load_train_feature_data(self, r_id, rule_list, train_id_data):
        if not os.path.exists(self.train_feature_data_file):
            train_x = self.get_features(rule_list, train_id_data[:, 0:-1])
            train_y = train_id_data[:, -1]

            train_x = np.array(train_x)

            if pca_or_cwa == "pca":
                nege_data = train_id_data[np.where(train_id_data[:, -1] == 0)]
                soft_label_list = self.ekg.get_every_triple_conf(nege_data[:], r_id)
                train_y = train_y.astype(np.float)
                train_y[np.where(train_id_data[:, -1] == 0)] = np.array(soft_label_list)

                train_x = train_x.astype(np.float)

            train_y = np.array(train_y).reshape((len(train_y), 1))
            np.save(self.train_feature_data_file, np.hstack((train_x, train_y)))
            return np.hstack((train_x, train_y))
        else:
            return np.load(self.train_feature_data_file)

    # '''
    # Test ability of Logistic Regression
    # Parameters:
    # -----------
    # tar_r_idx: int
    # id of target relatoin
    # test_model: LogisticRegression test_model
    # rule_list: list, [Rule(),Rule(),...]
    # A list of rule obejct, sorted by precision descendingly.
    # '''
    #
    # def test_model(self, tar_r_idx, model, rule_list, test_file_folder, record_file):
    #     print("Test test_model of Relation: {}.".format(self.r_id2name[tar_r_idx]))
    #     posi_ht_file = test_file_folder + "posi_ht.txt"
    #     nege_ht_file = test_file_folder + "nege_ht.txt"
    #
    #     if not os.path.exists(posi_ht_file) or not os.path.exists(nege_ht_file):
    #         posi_ht_list = self.r2ht[tar_r_idx]
    #         posi_ht_token = ["{},{}".format(ht_idx[0], ht_idx[-1]) for ht_idx in posi_ht_list]
    #
    #         if restrain_num_of_posis_neges and len(posi_ht_list) > restrain_num:
    #             posi_ht_list = random.sample(posi_ht_list, restrain_num)
    #             print("Restrain posi num from {} to {}.".format(
    #                 len(posi_ht_list), restrain_num))
    #
    #         nege_ht_list = []
    #
    #         e_idx_list = list(self.e_id2name.keys())
    #
    #         for ht_idx in posi_ht_list:
    #             h_idx = ht_idx[0]
    #             t_idx = ht_idx[-1]
    #             altered_idx_list = self.sample_from_list(e_idx_list, 20)
    #             cnt = 0
    #             for altered_e_idx in altered_idx_list:
    #                 if cnt >= 5:
    #                     break
    #                 token = "{},{}".format(h_idx, altered_e_idx)
    #                 if token not in posi_ht_token:
    #                     nege_ht_list.append([h_idx, altered_e_idx])
    #                     cnt += 1
    #             cnt = 0
    #             for altered_e_idx in list(reversed(altered_idx_list)):
    #                 if cnt >= 5:
    #                     break
    #                 token = "{},{}".format(altered_e_idx, t_idx)
    #                 if token not in posi_ht_token:
    #                     nege_ht_list.append([altered_e_idx, t_idx])
    #                     cnt += 1
    #
    #         # if restrain_num_of_posis_neges and len(nege_ht_list) > restrain_num:
    #         #     nege_ht_list = random.sample(nege_ht_list, restrain_num)
    #         #     print("Restrain posi num from {} to {}.".format(
    #         #         len(nege_ht_list), restrain_num))
    #
    #         # test_num = min(len(posi_ht_list), len(nege_ht_list))
    #         # posi_ht_list = random.sample(posi_ht_list, test_num)
    #         # nege_ht_list = random.sample(nege_ht_list, test_num)
    #
    #         print("Wring posi_ht and nege_ht to file.")
    #         with open(posi_ht_file, "w", encoding="UTF-8") as f:
    #             f.write("{}\n".format(len(posi_ht_list)))
    #             for ht in posi_ht_list:
    #                 f.write("{}\t{}\n".format(ht[0], ht[1]))
    #
    #         with open(nege_ht_file, "w", encoding="UTF-8") as f:
    #             f.write("{}\n".format(len(nege_ht_list)))
    #             for ht in nege_ht_list:
    #                 f.write("{}\t{}\n".format(ht[0], ht[1]))
    #
    #         # print(
    #         #     "Restrain num of posi/nege to {}. This is the minum length of posi_ht_list and nege_ht_list."
    #         #         .format(test_num))
    #     else:
    #         posi_ht_list = []
    #         nege_ht_list = []
    #         with open(posi_ht_file, 'r', encoding="UTF-8") as f:
    #             for idx, line in enumerate(f.readlines()):
    #                 if idx == 0:
    #                     continue
    #                 posi_ht_list.append([int(num) for num in line.split()])
    #         with open(nege_ht_file, 'r', encoding="UTF-8") as f:
    #             for idx, line in enumerate(f.readlines()):
    #                 if idx == 0:
    #                     continue
    #                 nege_ht_list.append([int(num) for num in line.split()])
    #
    #     print("Start getting features for positives.")
    #     test_x = self.get_features(rule_list, posi_ht_list)
    #     test_y = list(np.ones(len(posi_ht_list)))
    #
    #     print("Start getting features for negetives.")
    #     test_x.extend(self.get_features(rule_list, nege_ht_list))
    #     test_y.extend(list(np.zeros(len(nege_ht_list))))
    #
    #     precision = model.test_precision(test_x, test_y)
    #     print("Prec: {}".format(precision))
    #
    #     map_metric = model.test_map(test_x, test_y)
    #     print("MAP: {}".format(map_metric))
    #     with open(record_file, "a+", encoding="UTF-8") as f:
    #         f.write("Prec.: {}\n".format(precision))
    #         f.write("MAP: {}\n".format(map_metric))

    '''
    Sample from a list, if sampled num is larger than the size of the list,
    return the whole list.
    Parameters:
    -----------
    sampled_list: list
    a list to be sampled.
    sample_num: int
    the num we want to sample from sampled_list
    
    Returns:
    -----------
    out: list
    a sampled list
    '''

    def sample_from_list(self, sampled_list, sample_num):
        if len(sampled_list) <= sample_num:
            return sampled_list
        else:
            return list(random.sample(sampled_list, sample_num))

    """
    Get features for one pair of h and t
    Parameters:
    -----------
    rule_list: list, [Rule(),Rule(),Rule(),...]
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
        for ht_i, ht in enumerate(ht_list):
            if (ht_i + 1) % 100 == 0:
                print("Feature: {}/{}, #Rules: {}, H: {}, T: {}".format(
                    ht_i + 1, len(ht_list), len(rule_list), self.e_id2name[ht[0]],
                    self.e_id2name[ht[1]]))
            feature = []
            for rule in rule_list:
                feature.append(int(self.is_passed(ht, rule.get_r_id_rule())))
            train_x.append(feature)
        return train_x

    # """
    # Get top-K rules for a relation(r_idx) by criterion(P/R/F1)
    # Parameters:
    # -----------
    # r_idx: int
    # index for target relation
    # top_k: int
    # num of top rules we want to get
    # criterion: str
    # 'P', 'R' or 'F1', it is the criterion used to sort rule list
    #
    # Returns:
    # -----------
    # out: list, [Rule(),Rule(),Rule(),...]
    # top_k rules
    # """
    #
    # def get_top_k_rules(self, r_idx, top_k):
    #     rule_list = []
    #     query = "select relation_idx, rule_key from " + database + \
    #             " where relation_idx={} order by {} desc".format(r_idx, sort_rule_criterion)
    #     mycursor = mydb.cursor()
    #     mycursor.execute(query)
    #     fetched = mycursor.fetchall()
    #     fetched_num = len(fetched)
    #     for idx, row in enumerate(fetched):
    #         print("{}/{}".format(idx, fetched_num))
    #         if idx > top_k - 1:
    #             break
    #         one_rule = Rule(r_idx, None, row[1])
    #         one_rule.restoreFromMysql()
    #         rule_list.append(one_rule)
    #     return rule_list

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
                if r_idx not in self.r_id2name:
                    return False
                for e_idx in left_node:
                    c_node = self.node_dict[e_idx]
                    for tail in c_node.get_tails_of_r_idx(r_idx):
                        temp_left.append(tail)
                left_node = temp_left
            else:
                right_step += 1
                r_idx = rule[-right_step]
                if r_idx not in self.r_id2name:
                    return False
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
                    if ht_token in ht_is_passed:
                        continue
                    th = MyThread(self.is_passed, ([h_idx, t_idx], rule))
                    thread_list.append(th)
                    t_idx2ht_token[len(thread_list) - 1] = ht_token

            [th.start() for th in thread_list]
            [th.join() for th in thread_list]

            for th_idx in range(len(thread_list)):
                ht_token = t_idx2ht_token[th_idx]
                if thread_list[th_idx].get_result():
                    ht_is_passed[ht_token] = 1

        return [ht_token.split(ht_conn)
                for ht_token in ht_is_passed.keys()], set(ht_is_passed.keys())

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

    def get_ht_from_one_end(self, h_idx_list, t_idx_list, rule,
                            passed_ht_token_set):
        assert (len(h_idx_list) != 0
                or len(t_idx_list) != 0), "Can't get ht from one end."

        def ht2token(h, t):
            return ht_conn.join([str(h), str(t)])

        res_left_path_list = [[h_idx] for h_idx in h_idx_list]
        res_right_path_list = [[t_idx] for t_idx in t_idx_list]
        left_empty = True if len(h_idx_list) == 0 else False

        if not left_empty:
            for r_idx in rule:
                res_left_path_list = self.get_ht_path_from_left(
                    res_left_path_list, r_idx)
        else:
            for r_idx in list(reversed(rule)):
                res_right_path_list = self.get_ht_path_from_right(
                    res_right_path_list, r_idx)

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

    '''
    Get the e_r path between h_idx and t_idx, meanwhile, get the features for rule_list
    Parameters:
    -----------
    h_idx: int
    The id of head.
    t_idx: int
    The id of tail.
    rule_list: list [Rule(),Rule(),Rule(),....]
    A list of rules to check.
    
    Returns:
    ----------
    features: list
    Every entry indicates whether a rule is passed by h_idx and t_idx.
    For example, [0,1,0,0,...]
    path: list, [[-1,e,r,e,..,e,-1],[],[],....]
    Passed path for every rule.
    '''

    def get_passed_e_r_path(self, h_idx, t_idx, rule_list):
        features = []
        res_path = []
        # [res_path.append([]) for _ in rule_list]
        [features.append(0) for _ in rule_list]

        for rule_idx, rule_obj in enumerate(rule_list):
            rule_path = rule_obj.get_r_id_rule()
            left_path_queue = queue.Queue()
            right_path_queue = queue.Queue()
            left_path_queue.put([-1, h_idx])
            right_path_queue.put([-1, t_idx])
            left_step = 0
            right_step = 0
            while len(rule_path) - (left_step + right_step) > 0:
                left_size = left_path_queue.qsize()
                right_size = right_path_queue.qsize()
                if left_size < right_size:
                    left_step += 1
                    r_idx = rule_path[left_step - 1]
                    out_queue_cnt = 0
                    while not left_path_queue.empty(
                    ) and out_queue_cnt < left_size:
                        c_path = left_path_queue.get()
                        out_queue_cnt += 1
                        c_node = self.node_dict[c_path[-1]]
                        for tail in c_node.get_tails_of_r_idx(r_idx):
                            temp_path = c_path.copy()
                            temp_path.append(r_idx)
                            temp_path.append(tail)
                            left_path_queue.put(temp_path)
                else:
                    right_step += 1
                    r_idx = rule_path[-right_step]
                    inv_r_idx = self.convert_r(r_idx)
                    out_queue_cnt = 0
                    while not right_path_queue.empty(
                    ) and out_queue_cnt < right_size:
                        c_path = right_path_queue.get()
                        out_queue_cnt += 1
                        c_node = self.node_dict[c_path[-1]]
                        for tail in c_node.get_tails_of_r_idx(inv_r_idx):
                            temp_path = c_path.copy()
                            temp_path.append(r_idx)
                            temp_path.append(tail)
                            right_path_queue.put(temp_path)

            left_dict = {}
            while not left_path_queue.empty():
                one_left_path = left_path_queue.get()
                if one_left_path[-1] not in left_dict:
                    left_dict[one_left_path[-1]] = []
                left_dict[one_left_path[-1]].append(one_left_path)

            while not right_path_queue.empty():
                one_right_path = right_path_queue.get()
                c_node_idx = one_right_path[-1]
                if c_node_idx in left_dict:
                    features[rule_idx] = 1
                    for one_left_path in left_dict[c_node_idx]:
                        temp_path = one_left_path[:-1]
                        temp_path.extend(list(reversed(one_right_path)))
                        res_path.append(temp_path)
        return features, res_path

    '''
    Get r_name by r_idx
    Parameters:
    -----------
    r_idx: int
    the index of relation
    
    Returns:
    -----------
    out: string
    name of r_idx
    '''

    def get_r_name_by_r_idx(self, r_idx):
        assert r_idx in self.r_id2name, "R_idx is not in this graph."
        return self.r_id2name[r_idx]

    '''
    Get e_name by e_idx
    Parameters:
    -----------
    e_idx: int
    the index of entity

    Returns:
    -----------
    out: string
    name of e_idx
    '''

    def get_e_name_by_e_idx(self, e_idx):
        assert e_idx in self.e_id2name, "E_idx is not in this graph."
        return self.e_id2name[e_idx]

    '''
    Get e_idx by e_name
    Parameters:
    -----------
    e_name: string
    the name of entity

    Returns:
    -----------
    out: int
    idx of e_name
    '''

    def get_e_idx_by_e_name(self, e_name):
        assert e_name in self.e_name2id, "E_name is not in this graph"
        return self.e_name2id[e_name]

    '''
    Get r_idx by r_name
    Parameters:
    -----------
    r_name: string
    the name of relation
    
    Parameters:
    -----------
    out: int
    idx of r_name
    '''

    def r_id_by_name(self, r_name):
        assert r_name in self.r_name2id, "E_name is not in this graph"
        return self.r_name2id[r_name]

    '''
    Get localname of e/r name
    Parameters:
    -----------
    e_r_name: string
    For example, dbo:United_States (only support this form)
    Returns:
    -----------
    out: string
    localname of e/r name
    To example, the localname of dbo:United_States is United_States
    '''

    def get_localname(self, e_r_name):
        return e_r_name.split(":")[-1]

    #
    # '''
    # Get mined rules of relation
    # Parameters:
    # -------------
    # r_name: string, the relation we want to get the rules
    # Returns:
    # None
    # It will print the rules per line.
    # '''
    #
    # def get_rules_4_relation(self, r_name):
    #     if r_name not in self.r2idx:
    #         print("R: {} is not in the Graph.".format(r_name))
    #         return
    #     print("Get rules of R: {}.".format(r_name))
    #     temp_idx = self.r2idx[r_name]
    #     query = "SELECT rule_key,P FROM `{}` where relation_idx='{}'".format(database, temp_idx)
    #     mycursor = mydb.cursor()
    #     mycursor.execute(query)
    #     fetched = mycursor.fetchall()
    #     res = {}
    #     for row in fetched:
    #         my_rule = row[0]
    #         rule_fetched = ",\t".join([self.idx2r[int(one_r)] for one_r in my_rule.split(":")])
    #         res[rule_fetched] = float(row[1])
    #     a = list(sorted(res.items(), key=lambda x: x[1], reverse=True))
    #     for key, value in a:
    #         print(key, value)

    """
    Get searched result from head_idx by r_rule_list, and get prob by r_mode
    search is from left to right (head to tail).
    Parameters:
    -----------
    head_idx: int, the idx of head
    r_rule_list: list of Rule object, [Rule(),Rule(),...]
    r_model: classifier old_model
    
    Return:
    -----------
    out: [[t_idx,prob],[],[],...]
    """

    def get_tail_from_head_by_rule(self, head_idx, r_rule_list, r_model):
        t_rule_dict = {}
        t_prob_list = []
        # print("Head: {}".format(self.idx2e[head_idx]))
        for rule_idx, rule in enumerate(r_rule_list):
            # print("Rule: {}".format(self.display_r_path([rule.r_path])))
            h_node_queue = queue.Queue()
            h_node_queue.put(head_idx)
            for rule_r_idx in rule.get_r_id_rule():
                # print("R: {}".format(self.idx2r[rule_r_idx]))
                tmp_res = []
                while not h_node_queue.empty():
                    c_head_idx = h_node_queue.get()
                    # print("E: {}".format(self.idx2e[c_head_idx]))
                    c_node = self.node_dict[c_head_idx]
                    tmp_res.extend(c_node.get_r_value(rule_r_idx))
                [h_node_queue.put(e_idx) for e_idx in list(set(tmp_res))]

            passed_ht_token = set()
            while not h_node_queue.empty():
                e_idx = h_node_queue.get()
                passed_ht_token.add(e_idx)

            for token in passed_ht_token:
                if token not in t_rule_dict:
                    t_rule_dict[token] = []

                t_rule_dict[token].append(rule_idx)

        for token in t_rule_dict.keys():
            one_feature = [0] * len(r_rule_list)
            for rule_idx in t_rule_dict[token]:
                one_feature[rule_idx] = 1
            prob = r_model.get_output_prob(one_feature)
            t_prob_list.append([token, prob])
        t_prob_list.sort(key=lambda x: x[1], reverse=True)
        return t_prob_list

    """
    Get searched result from head_idx by r_rule_list, and get prob by r_mode
    search is from left to right (tail to head).
    Parameters:
    -----------
    tail_idx: int, the idx of head
    r_rule_list: list of Rule object, [Rule(),Rule(),...]
    r_model: classifier old_model

    Return:
    -----------
    out: [[h_idx,prob],[],[],...]
    """

    def get_head_from_tail_by_rule(self, tail_idx, r_rule_list, r_model):
        h_rule_dict = {}
        h_prob_list = []
        # print("Head: {}".format(self.idx2e[head_idx]))
        for rule_idx, rule in enumerate(r_rule_list):
            # print("Rule: {}".format(self.display_r_path([rule.r_path])))
            h_node_queue = queue.Queue()
            h_node_queue.put(tail_idx)
            for rule_r_idx in rule.get_r_id_rule():
                rule_r_name = self.r_id2name[rule_r_idx]
                inv_rule_r_name = self.util.get_inverse_r(rule_r_name)
                inv_rule_r_idx = self.r_name2id[inv_rule_r_name]
                # print("R: {}".format(self.idx2r[rule_r_idx]))
                tmp_res = []
                while not h_node_queue.empty():
                    c_head_idx = h_node_queue.get()
                    # print("E: {}".format(self.idx2e[c_head_idx]))
                    c_node = self.node_dict[c_head_idx]
                    tmp_res.extend(c_node.get_r_value(inv_rule_r_idx))
                [h_node_queue.put(e_idx) for e_idx in list(set(tmp_res))]

            passed_ht_token = set()
            while not h_node_queue.empty():
                e_idx = h_node_queue.get()
                # token = "{},{}".format(e_idx, tail_idx)
                passed_ht_token.add(e_idx)

            for token in passed_ht_token:
                if token not in h_rule_dict:
                    h_rule_dict[token] = []

                h_rule_dict[token].append(rule_idx)

        for token in h_rule_dict.keys():
            one_feature = [0] * len(r_rule_list)
            for rule_idx in h_rule_dict[token]:
                one_feature[rule_idx] = 1
            prob = r_model.get_output_prob(one_feature)
            h_prob_list.append([token, prob])
        h_prob_list.sort(key=lambda x: x[1], reverse=True)
        return h_prob_list

# if __name__ == "__main__":
#     dbpedia_scope = "All"
#     e2idx_file = dbpedia_folder + dbpedia_scope + file_path_seg + "e2idx_shortcut.txt"
#     r2idx_file = dbpedia_folder + dbpedia_scope + file_path_seg + "r2idx_shortcut.txt"
#     triple2idx_file = dbpedia_folder + dbpedia_scope + file_path_seg + "triple2idx.txt"
#     graph = Graph(e2idx_file, r2idx_file, triple2idx_file)
#     graph.load_data()
#     print("")
#     # graph.get_rules_4_relation("dbo:birthPlace")
#     # graph.get_rules_4_relation("dbo:regionServed")
#     # graph.get_rules_4_relation("dbo:location")
#     # graph.get_rules_4_relation("dbo:owner")
#     # graph.get_rules_4_relation("dbo:founder")
#     # graph.get_rules_4_relation("dbo:foundationPlace")
#     # graph.get_rules_4_relation("dbo:residence")
#     # graph.get_rules_4_relation("dbo:headquarter")
#     # graph.get_rules_4_relation("dbo:residence")
#     # graph.get_rules_4_relation("dbo:locationCountry")
#
#     # util = Util()
#     # source_folder = "./source/"
#     # source_folder = "F:\\Data\\FB15K-237\\source\\"
#     # output_folder = "F:\\Data\\FB15K-237\\output\\"
#     # e2idx_file = source_folder + "e2idx.txt"
#     # r2idx_file = source_folder + "r2idx.txt"
#     # triple2idx_file = source_folder + "triple2idx.txt"
#
#     # relation_list = ['<http://dbpedia.org/ontology/director>',
#     #                  '<http://dbpedia.org/ontology/starring>',
#     #                  '<http://dbpedia.org/ontology/birthPlace>']
#
#     # relation_list = ['/film/film/language']
#     # graph = Graph(e2idx_file, r2idx_file, triple2idx_file)
#     # graph.load_data()
#
#     # r_idx_list = [graph.r2idx[relation] for relation in relation_list]
#     # r_rules_dict = {}
#     # for idx, r_idx in enumerate(r_idx_list):
#     #     folder = output_folder + util.gen_prefix(
#     #         relation_list[idx]) + file_path_seg
#     #     if not os.path.isdir(folder):
#     #         os.makedirs(folder)
#     #     graph.get_pra_model4r(r_idx=r_idx, folder=folder)
#     #     r_rules_dict[r_idx] = graph.get_top_k_rules(r_idx, 5, 'P')
#     #
#     # for key in r_rules_dict.keys():
#     #     print("Relation: {}".format(graph.idx2r[key]))
#     #     for p in graph.display_r_path(r_rules_dict[key]):
#     #         print("=>".join(p))
#     #     print("\n")
#
#     # displayed_path = graph.display_r_path(r_path_list)
#     # displayed_path = graph.display_e_r_path(e_r_path_list)
#     # for p in displayed_path:
#     #     print("=>".join(p))
