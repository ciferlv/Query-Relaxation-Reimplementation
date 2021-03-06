# import mysql.connector

from RuleBased.Params import rule_seg, rule_num2display, pca_or_cwa

from Util import Util


class Path:
    def __init__(self, r, e):
        self.r = int(r)
        self.e = int(e)

    def __eq__(self, other):
        return int(self.r) == int(other.r) and int(self.e) == int(other.e)


class Node:
    def __init__(self, e_key):
        self.e_key = int(e_key)
        self.path_list = []
        self.path_dict = {}

    def get_r_value(self, r_idx):
        if r_idx not in self.path_dict:
            return []
        else:
            return self.path_dict[r_idx]

    def addPath(self, r, e):
        r = int(r)
        e = int(e)
        self.path_list.append(Path(r=r, e=e))
        if r not in self.path_dict:
            self.path_dict[r] = []
        self.path_dict[r].append(e)

    def get_tails_of_r_idx(self, r_idx):
        if r_idx not in self.path_dict:
            return []
        return self.path_dict[r_idx]
        # tail_list = []
        # for p in self.path_list:
        #     if int(p.r) == int(r_idx):
        #         tail_list.append(p.e)
        # return tail_list

    def has_r(self, r_idx):
        return r_idx in self.path_dict

    def has_r_t(self, r_idx, t_idx):
        if r_idx not in self.path_dict:
            return False, False
        else:
            return True, t_idx in self.path_dict[r_idx]


class Rule:
    def __init__(self, r_id):
        self.r_id = r_id

        self.r_id_rule = []
        self.r_name_rule = []
        self.rule_id_key = None
        self.rule_name_key = None
        self.rule_len = 0

        self.passHT = []
        self.P = 0
        self.R = 0
        self.F1 = 0
        self.correct_ht = []
        self.wrong_ht = []
        self.no_idea_ht = []

        self.util = Util()

    def get_r_id_rule(self):
        return self.r_id_rule

    def set_r_id_rule(self, r_id_rule):
        self.r_id_rule = r_id_rule
        self.rule_id_key = rule_seg.join(map(str, r_id_rule))
        self.rule_len = len(self.r_id_rule)

    def set_r_name_rule(self, r_name_rule):
        self.r_name_rule = r_name_rule
        self.rule_name_key = rule_seg.join(self.r_name_rule)

    def set_P_R_F1(self, p, r, f1):
        self.P = p
        self.R = r
        self.F1 = f1

    def toStr(self):
        return "{}\t{}\t{}\t{}\t{}".format(self.rule_id_key, self.rule_name_key, self.P, self.R, self.F1)

    def get_P_R_F1(self, node_dict, r2ht):
        correct_add = 0.0
        for ht in self.passHT:
            has_r, has_r_t = node_dict[ht[0]].has_r_t(self.r_id, ht[-1])
            if has_r_t:
                self.correct_ht.append(ht)
                correct_add += 1
            elif has_r and not has_r_t:
                self.wrong_ht.append(ht)
            else:
                self.no_idea_ht.append(ht)

        self.R = correct_add / len(r2ht[self.r_id])

        if pca_or_cwa == "pca":
            self.P = correct_add / (len(self.passHT) - len(self.no_idea_ht))
        else:
            self.P = correct_add / len(self.passHT)

        # assert self.P != 0 and self.R != 0, "P R F1 has wrong calculation"
        if self.R != 0 or self.P != 0:
            self.F1 = 2 * self.R * self.P / (self.P + self.R)
        print("Prec. {}, Rec. {}, F1 {}.".format(self.P, self.R, self.F1))

    # def sample_ht(self, ht_list, sampled_num):
    #     if sampled_num < len(ht_list):
    #         sampled_ht = random.sample(ht_list, sampled_num)
    #     else:
    #         sampled_ht = ht_list
    #     return sampled_ht
    #
    # def persist2mysql(self):
    #     sampled_num = 1000
    #     self.correct_ht = self.sample_ht(self.correct_ht, sampled_num)
    #     self.wrong_ht = self.sample_ht(self.wrong_ht, sampled_num)
    #     self.no_idea_ht = self.sample_ht(self.no_idea_ht, sampled_num)
    #
    #     correct_ht_str = ht_seg.join(
    #         [ht_conn.join(map(str, ht)) for ht in self.correct_ht])
    #     wrong_ht_str = ht_seg.join(
    #         [ht_conn.join(map(str, ht)) for ht in self.wrong_ht])
    #     no_idea_ht_str = ht_seg.join(
    #         [ht_conn.join(map(str, ht)) for ht in self.no_idea_ht])
    #
    #     query = "INSERT INTO " + database + \
    #             "  ( relation_idx,rule_key,rule_len,correct_ht,wrong_ht,no_idea_ht,P,R,F1,rule4train) " \
    #             "VALUES ({},'{}',{},'{}','{}','{}',{},{},{},-1);" \
    #                 .format(self.r_idx, self.rule_key, self.rule_len,
    #                         correct_ht_str, wrong_ht_str,
    #                         no_idea_ht_str, self.P, self.R, self.F1)
    #     mycursor = mydb.cursor()
    #     try:
    #         mycursor.execute(query)
    #         mydb.commit()
    #         return True
    #     except Exception as e:
    #         print("Exception:{}\nInsert Failed, start rolling back.".format(e))
    #         mydb.rollback()
    #         return False

    # def restoreFromMysql(self):
    #     query = "select * from " + database + \
    #             " where relation_idx = {} and rule_key = '{}';".format(self.r_idx, self.rule_key)
    #     mycursor = mydb.cursor()
    #     mycursor.execute(query)
    #     fetched = mycursor.fetchall()
    #     if len(fetched) == 0:
    #         return False
    #     for row in fetched:
    #         self.rule_len = int(row[3])
    #         if len(row[4]) != 0:
    #             self.correct_ht = [
    #                 list(map(int, ht2)) for ht2 in
    #                 [ht.split(ht_conn) for ht in row[4].split(ht_seg)]
    #             ]
    #         if len(row[5]) != 0:
    #             self.wrong_ht = [
    #                 list(map(int, ht2)) for ht2 in
    #                 [ht.split(ht_conn) for ht in row[5].split(ht_seg)]
    #             ]
    #         if len(row[6]) != 0:
    #             self.no_idea_ht = [
    #                 list(map(int, ht2)) for ht2 in
    #                 [ht.split(ht_conn) for ht in row[6].split(ht_seg)]
    #             ]
    #         self.P = row[7]
    #         self.R = row[8]
    #         self.F1 = row[9]
    #     return True

    # """
    # Sample positive OneCons_eval and negetive OneCons_eval to train
    # Parameters:
    # -----------
    # posi_num: sampled num for positive OneCons_eval
    # nege_num： sampled num for negetive OneCons_eval
    #
    # Retures:
    # -----------
    # positives: list, [[h_idx,t_idx],[h_idx,t_idx],[h_idx,t_idx],...]
    #            the sampled positive OneCons_eval
    # negetives: list, [[h_idx,t_idx],[h_idx,t_idx],[h_idx,t_idx],...]
    #            the sampled negetive OneCons_eval
    # """
    #
    # def sample_train_data(self, posi_num, nege_num):
    #     sampled_correct_ht = []
    #     sampled_wrong_ht = []
    #     assert len(self.correct_ht) != 0 or len(
    #         self.wrong_ht) != 0, "Haven't load correct/wrong ht"
    #     if posi_num > len(self.correct_ht):
    #         sampled_correct_ht.extend(self.correct_ht)
    #     else:
    #         sampled_correct_ht.extend(random.sample(self.correct_ht, posi_num))
    #     if nege_num > len(self.wrong_ht):
    #         sampled_wrong_ht.extend(self.wrong_ht)
    #     else:
    #         sampled_wrong_ht.extend(random.sample(self.wrong_ht, nege_num))
    #     return sampled_correct_ht, sampled_wrong_ht

    # def add_train_label(self, label):
    #     query = "UPDATE " + database + "  SET rule4train = {} where relation_idx = {} and rule_key = '{}';" \
    #         .format(label, self.r_idx, self.rule_key)
    #     mycursor = mydb.cursor()
    #     try:
    #         mycursor.execute(query)
    #         mydb.commit()
    #         print(
    #             "Success updating train label for R: {}, Rule Key: {}.".format(
    #                 self.r_idx, self.rule_key))
    #         return True
    #     except Exception as e:
    #         assert False, "Exception:{}\nUpdate Failed, start rolling back.".format(
    #             e)
    #         mydb.rollback()
    #         return False


class Candidate:
    def __init__(self, candidate_list):
        self.candidate_list = candidate_list
        self.body_bgp = []
        self.pra_conf = {}
        self.embedding_conf = {}
        self.bgp_path_dict = {}
        for one_bgp_list in self.body_bgp:
            self.bgp_path_dict["\t".join(one_bgp_list)] = []

        self.cand_pra_score = 0
        self.cand_embedding_score = 0

    '''
    Multiply every bgp's pra confidence to get confidence of the whole candidate
    '''

    def cal_cand_pra_score(self):
        res_score = 1
        for key in self.pra_conf:
            res_score *= self.pra_conf[key]
        self.cand_pra_score = res_score

    # '''
    # Multiply every bgp's embedding confidence to get confidence of the whole candidate
    # '''
    #
    # def cal_can_embedding_score(self):
    #     res_score = 1
    #     for key in self.embedding_conf:
    #         res_score *= self.embedding_conf[key]
    #     self.cand_embedding_score = res_score

    '''
    Return total conf of a candidate
    if sort_candidate_criterion is pra, return pra score
    else return embedding score
    ----------------
    Parameters:
    sort_way: pra or embedding
    '''

    def get_conf(self, sort_way):
        if sort_way == "pra":
            return self.cand_pra_score
        else:
            return self.cand_embedding_score

    '''
    Add a bgp with variable filled.
    Parameters:
    ------------
    one_bgp_str: string
    for example, h_name /t r_name /t t_name
    '''

    def add_completed_bgp(self, one_bgp_str):
        self.body_bgp.append(one_bgp_str)

    '''
    Add Path passed for one bgp
    Parameters:
    -----------
    path_list: list, [[-1,e,r,e,..,e,-1],[],[],....]
    one_bgp_str: string, for example, h_name /t r_name /t t_name
    '''

    def add_path_idx_for_bgp(self, path_list, one_bgp_str):
        self.bgp_path_dict[one_bgp_str] = path_list

    '''
    Add confidence calculated by pra for one bgp
    Parameters:
    -----------
    pra_conf: float
    one_bgp_str: string, for example, h_name /t r_name /t t_name
    '''

    def add_pra_conf_for_bgp(self, pra_conf, one_bgp_str):
        self.pra_conf[one_bgp_str] = pra_conf

    # '''
    # Add  confidence calculated by embedding for one bgp
    # Parameters:
    # -----------
    # h_vec: list, vector for head, [-1,2,0.5,..,3.2,5,..]
    # r_vec: list, vector for relation, [-1,2,0.5,..,3.2,5,..]
    # t_vec: list, vector for tail, [-1,2,0.5,..,3.2,5,..]
    # one_bgp_str: string, for example, h_name /t r_name /t t_name
    # '''
    #
    # def get_embedding_conf_for_bgp(self, h_vec, r_vec, t_vec, one_bgp_str):
    #     temp_res = h_vec + r_vec
    #     cosin_score = np.matmul(temp_res, t_vec) / (np.linalg.norm(temp_res) * np.linalg.norm(t_vec))
    #     self.embedding_conf[one_bgp_str] = cosin_score

    def display_rule_path(self, graph):
        res_str = ""
        for one_bgp_str in self.body_bgp:
            res_str += "{} pra_conf: {}\n".format(one_bgp_str, self.pra_conf[one_bgp_str])

            for idx, one_path in enumerate(
                    graph.display_e_r_path(self.bgp_path_dict[one_bgp_str])):
                if idx >= rule_num2display:
                    break
                res_str += "{}\n".format(rule_seg.join(one_path))
            res_str += "\n"
        return res_str.strip()

    def display_var2entity(self, var_list, graph):
        res_str = ""
        for var_idx, var_name in enumerate(var_list):
            res_str += "{}:[{}]\t".format(
                var_name,
                graph.get_e_name_by_e_idx(self.candidate_list[var_idx][0]))
        return res_str.strip()
