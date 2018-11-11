import mysql.connector

from RuleBased.Params import ht_seg, ht_conn, mydb, database, rule_seg
import random


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

    def addPath(self, r, e):
        self.path_list.append(Path(r=r, e=e))

    def get_tails_of_r_idx(self, r_idx):
        tail_list = []
        for p in self.path_list:
            if int(p.r) == int(r_idx):
                tail_list.append(p.e)

        return tail_list

    def has_r(self, r_idx):
        for p in self.path_list:
            if r_idx == p.r: return True
        return False

    def has_r_t(self, r_idx, t_idx):
        has_r = False
        has_r_t = False
        for p in self.path_list:
            if p.r == r_idx:
                has_r = True
                if p.e == t_idx:
                    has_r_t = True
                    break
        return has_r, has_r_t


class Rule:
    def __init__(self, r_idx, r_path=None, rule_key=None):
        self.r_idx = r_idx
        if rule_key is None:
            self.rule_key = ":".join(map(str, r_path))
            self.r_path = r_path
        elif r_path is None:
            self.rule_key = rule_key.strip()
            self.r_path = list(map(int, self.rule_key.split(rule_seg)))
        else:
            self.rule_key = rule_key
            self.r_path = r_path

        self.rule_len = len(self.r_path)
        self.passHT = []
        self.P = 0
        self.R = 0
        self.F1 = 0
        self.correct_ht = []
        self.wrong_ht = []
        self.no_idea_ht = []

    def get_P_R_F1(self, node_dict, r2ht):
        for ht in self.passHT:
            has_r, has_r_t = node_dict[ht[0]].has_r_t(self.r_idx, ht[-1])
            if has_r_t:
                self.correct_ht.append(ht)
            elif has_r and not has_r_t:
                self.wrong_ht.append(ht)
            else:
                self.no_idea_ht.append(ht)

        self.R = len(self.correct_ht) / len(r2ht[self.r_idx])
        self.P = len(self.correct_ht) / len(self.passHT)
        self.F1 = 2 * self.R * self.P / (self.P + self.R)
        assert self.P != 0 and self.R != 0, "P R F1 has wrong calculation"

    def sample_ht(self, ht_list, sampled_num):
        if sampled_num < len(ht_list):
            sampled_ht = random.sample(ht_list, sampled_num)
        else:
            sampled_ht = ht_list
        return sampled_ht

    def persist2mysql(self):
        sampled_num = 1000
        self.correct_ht = self.sample_ht(self.correct_ht, sampled_num)
        self.wrong_ht = self.sample_ht(self.wrong_ht, sampled_num)
        self.no_idea_ht = self.sample_ht(self.no_idea_ht, sampled_num)

        correct_ht_str = ht_seg.join([ht_conn.join(map(str, ht)) for ht in self.correct_ht])
        wrong_ht_str = ht_seg.join([ht_conn.join(map(str, ht)) for ht in self.wrong_ht])
        no_idea_ht_str = ht_seg.join([ht_conn.join(map(str, ht)) for ht in self.no_idea_ht])

        query = "INSERT INTO" + database + "  ( relation_idx,rule_key,rule_len,correct_ht,wrong_ht,no_idea_ht,P,R,F1 ) VALUES ({},'{}',{},'{}','{}','{}',{},{},{});" \
            .format(self.r_idx, self.rule_key, self.rule_len, correct_ht_str, wrong_ht_str,
                    no_idea_ht_str, self.P, self.R, self.F1)
        mycursor = mydb.cursor()
        try:
            mycursor.execute(query)
            mydb.commit()
            return True
        except Exception as e:
            print("Exception:{}\nInsert Failed, start rolling back.".format(e))
            mydb.rollback()
            return False
        mydb.close()

    def restoreFromMysql(self):
        query = "select * from" + database + \
                " where relation_idx = {} and rule_key = '{}';".format(self.r_idx, self.rule_key)
        mycursor = mydb.cursor()
        mycursor.execute(query)
        fetched = mycursor.fetchall()
        assert len(fetched) <= 1, "Duplicate relation:rulepath in MYSQL."
        if len(fetched) == 0:
            return False
        for row in fetched:
            self.rule_len = int(row[3])
            self.correct_ht = [list(map(int, ht2)) for ht2 in [ht.split(ht_conn) for ht in row[4].split(ht_seg)]]
            self.wrong_ht = [list(map(int, ht2)) for ht2 in [ht.split(ht_conn) for ht in row[5].split(ht_seg)]]
            self.no_idea_ht = [list(map(int, ht2)) for ht2 in [ht.split(ht_conn) for ht in row[6].split(ht_seg)]]
            self.P = row[7]
            self.R = row[8]
            self.F1 = row[9]
        return True

    """
    Sample positive data and negetive data to train
    Parameters:
    -----------
    posi_num: sampled num for positive data
    nege_num： sampled num for negetive data
    
    Retures:
    -----------
    positives: list 
               the sampled positive data
    negetives: list
               the sampled negetive data
    """

    def sample_train_data(self, posi_num, nege_num):
        assert len(self.correct_ht) != 0 and len(self.wrong_ht) != 0, "Haven't load correct/wrong ht"
        if posi_num > len(self.correct_ht):
            sampled_correct_ht = self.correct_ht
        else:
            sampled_correct_ht = random.sample(self.correct_ht, posi_num)
        if nege_num > len(self.wrong_ht):
            sampled_wrong_ht = self.wrong_ht
        else:
            sampled_wrong_ht = random.sample(self.wrong_ht, nege_num)
        return sampled_correct_ht, sampled_wrong_ht
    #
    # """
    # Test if a ht is in this rule's correct_ht_list.
    # Parameters:
    # -----------
    # ht: list
    # a list of two length, for example, [head,tail]
    #
    # Returns:
    # -----------
    # out: boolean
    # If this ht is in correct_ht.
    # """
    #
    # def is_correct_ht(self, ht):
    #     for c_ht in self.correct_ht:
    #         if c_ht[0] == ht[0] and c_ht[1] == ht[1]:
    #             return True
    #     return False
    #
    # """
    # Test if a ht is in this rule's wrong_ht_list.
    # Parameters:
    # -----------
    # ht: list
    # a list of two length, for example, [head,tail]
    #
    # Returns:
    # -----------
    # out: boolean
    # True if this ht is in wrong_ht_list, False otherwise.
    # """
    #
    # def is_wrong_ht(self, ht):
    #     for w_ht in self.wrong_ht:
    #         if w_ht[0] == ht[0] and w_ht[1] == ht[1]:
    #             return True
    #     return False
    #
    # """
    # Test if a ht is in this rule's no_idea_ht_list.
    # Parameters:
    # -----------
    # ht: list
    # a list of two length, for example, [head,tail]
    #
    # Returns:
    # -----------
    # out: boolean
    # True if this ht is in no_idea_ht_list, False otherwise.
    # """
    #
    # def is_no_idea_ht(self, ht):
    #     for n_ht in self.no_idea_ht:
    #         if n_ht[0] == ht[0] and n_ht[1] == ht[1]:
    #             return True
    #     return False
    #
    # '''
    # Test if a ht can pass the rule
    # Parameters:
    # ------------
    # ht: list
    # a list of two length, for example, [head,tail]
    #
    # Returns:
    # -----------
    # out: boolean
    # If this ht can pass the rule.
    # '''
    #
    # def is_passed_ht(self, ht):
    #     if self.is_correct_ht(ht): return True
    #     if self.is_wrong_ht(ht): return True
    #     if self.is_no_idea_ht(ht): return True
    #     return False
