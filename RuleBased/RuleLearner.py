from SPARQLWrapper import SPARQLWrapper, JSON
import random
import math
import os
import queue
import numpy as np
from sklearn.model_selection import train_test_split

from RuleBased.ALogger import ALogger
from RuleBased.Classifier import LogisticRegression
from RuleBased.Util import Util
from RuleBased.unit.Rule import Rule
from RuleBased.unit.Triple import Triple

sparql_database = "http://210.28.132.61:8898/sparql"


class RuleLearner:
    def __init__(self, predicate, positive_num, rules_top_k):
        self.logger = ALogger("RuleLearner.py", True).getLogger()
        self.utils = Util()
        self.predicate = predicate
        self.sparql = SPARQLWrapper(sparql_database)
        self.positive_instance_list = []
        self.negetive_instance_list = []
        self.raw_rule_list = []
        self.checked_rule_list = []
        self.filtered_rule_sorted_by_accuracy_list = []
        self.filtered_rule_sorted_by_recall_list = []
        self.rules_top_k = rules_top_k
        self.positive_num = positive_num
        self.parsed_rule_sorted_by_accuracy_list = {}

        self.folder = "./data/" + self.utils.gen_prefix(self.predicate) + "/"
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        self.positive_file_path = self.folder + "positive_instances.txt"
        self.negetive_file_path = self.folder + "negetive_instances.txt"
        self.raw_rule_file = self.folder + "raw_rule.txt"
        self.checked_rule_file = self.folder + "checked_rule.txt"
        self.filtered_rule_sorted_by_accuracy_file = self.folder + "filtered_rule_sorted_by_accuracy.txt"
        self.filtered_rule_sorted_by_recall_file = self.folder + "filterd_rule_sorted_by_recall.txt"
        self.positive_features_path = self.folder + "positive_features.npy"
        self.negetive_features_path = self.folder + "negetive_features.npy"
        self.model_path = self.folder + "model.tar"

    def calculate_search_num_per_time(self, total_num, positive_num):
        max_num_per_time = 10000
        if total_num < positive_num:
            return 1, total_num
        elif positive_num < total_num < max_num_per_time:
            return 1, positive_num
        else:
            search_times = int(total_num / max_num_per_time)
            num_per_time = math.ceil(positive_num / search_times)
            return search_times, num_per_time

    def load_positive_instances(self):
        if len(self.positive_instance_list) != 0: return
        with open(self.positive_file_path, "r", encoding="UTF-8") as f:
            for idx, line in enumerate(f.readlines()):
                if idx == 0:
                    continue
                else:
                    subj = line.strip().split()[0]
                    obj = line.strip().split()[1]
                    self.positive_instance_list.append(Triple(None, subj, obj))

    def crawl_positive_instances(self):
        triple_pattern = "?s " + self.predicate + " ?o"
        res_num = self.utils.get_num_by_sparql("""select count(?s) where {""" + triple_pattern + """}""")
        search_times, num_per_time = self.calculate_search_num_per_time(total_num=res_num,
                                                                        positive_num=self.positive_num)

        print("Search_times: {}.".format(search_times))
        print("Num_per_time: {}.".format(num_per_time))

        for i in range(search_times):
            sparql_query = "select ?s ?o where {" + triple_pattern + "} limit 10000  offset " + str(10000 * i)
            self.logger.info(sparql_query)
            self.sparql.setQuery(sparql_query)
            self.sparql.setReturnFormat(JSON)
            results = self.sparql.query().convert()
            res_list = results['results']['bindings']
            for sampled_num in random.sample(range(0, 10000), num_per_time):
                self.positive_instance_list.append(Triple(res_list[sampled_num], None, None))
        with open(self.positive_file_path, "w") as f:
            f.write("Num: {}\n".format(len(self.positive_instance_list)))
            for instance in self.positive_instance_list:
                f.write("{}\n".format(instance))

    def get_posi_instances(self):
        if len(self.positive_instance_list) != 0: return
        if os.path.exists(self.positive_file_path):
            self.load_positive_instances()
        else:
            self.crawl_positive_instances()

    def load_raw_rule(self):
        if len(self.raw_rule_list) == 0:
            with open(self.raw_rule_file, "r", encoding="UTF-8") as f:
                for line in f.readlines():
                    self.raw_rule_list.append(line)

    def get_raw_rule(self):
        if os.path.exists(self.raw_rule_file):
            self.load_raw_rule()
        else:
            for i, instance in enumerate(self.positive_instance_list):
                self.logger.info("No.{}".format(i))
                instance.search_rule()
                self.raw_rule_list.extend(list(instance.rule_set))
            self.raw_rule_list = list(set(self.raw_rule_list))
            with open(self.raw_rule_file, "w") as f:
                for rule in self.raw_rule_list:
                    f.write(rule + "\n")

    def load_checked_rule(self):
        if len(self.checked_rule_list) != 0: return
        with open(self.checked_rule_file, "r") as f:
            for line in f.readlines():
                rule_chain, accuracy, recall, f1 = line.strip().split("\t")
                self.checked_rule_list.append(Rule(rule_chain, float(accuracy), float(recall), float(f1)))

    def check_raw_rule(self):
        if not os.path.exists(self.checked_rule_file):
            total_num_query = "select count(?s) where {?s " + self.predicate + " ?e.}"
            total_num = self.utils.get_num_by_sparql(total_num_query)
            for idx, raw_rule in enumerate(self.raw_rule_list):
                self.logger.info("No.{} {}".format(idx, raw_rule.strip()))
                if raw_rule.strip().split(";")[0][1:] == self.predicate[1:-1]:
                    continue

                triple_pattern = self.utils.rule_parser(raw_rule)
                query1 = "select count(?s) where{ " + triple_pattern + "FILTER EXISTS { ?s " + self.predicate + " ?random.}}"
                query2 = "select count(?s)where{ " + triple_pattern + "?s " + self.predicate + " ?e.}"

                correct_num = self.utils.get_num_by_sparql(query2)
                retrived_num = self.utils.get_num_by_sparql(query1)

                accuracy = correct_num * 1.0 / retrived_num
                recall = correct_num / total_num
                f1 = 2 * accuracy * recall / (accuracy + recall)

                self.checked_rule_list.append(Rule(raw_rule, accuracy, recall, f1))

            with open(self.checked_rule_file, "w") as f:
                for one_checked_rule in self.checked_rule_list:
                    f.write("{}\n".format(one_checked_rule))

    def load_filtered_rule_sorted_accuracy(self):
        if len(self.filtered_rule_sorted_by_accuracy_list) != 0: return
        with open(self.filtered_rule_sorted_by_accuracy_file, "r", encoding="UTF-8") as f:
            for line in f.readlines():
                rule_chain, accuracy, recall, f1 = line.strip().split("\t")
                self.filtered_rule_sorted_by_accuracy_list.append(
                    Rule(rule_chain, float(accuracy), float(recall), float(f1)))

    # filter rule by accuracy and recall, sorte rule by accuracy
    def filtered_sort_checked_rule(self):
        if not os.path.exists(self.filtered_rule_sorted_by_accuracy_file):
            filtered_rule_heap = queue.PriorityQueue()
            with open(self.checked_rule_file, "r") as f:
                for line in f.readlines():
                    rule_chain, accuracy, recall, f1 = line.strip().split("\t")
                    if float(accuracy) < 0.001 or float(recall) < 0.001:
                        continue
                    filtered_rule_heap.put(
                        Rule(rule_chain, float(accuracy), float(recall), float(f1)))
            with open(self.filtered_rule_sorted_by_accuracy_file, "w") as f:
                while not filtered_rule_heap.empty():
                    f.write("{}\n".format(filtered_rule_heap.get()))

    def load_negetive_instances(self):
        if len(self.negetive_instance_list) != 0: return
        with open(self.negetive_file_path, "r", encoding="UTF-8") as f:
            for idx, line in enumerate(f.readlines()):
                if idx == 0:
                    continue
                else:
                    subj = line.strip().split()[0]
                    obj = line.strip().split()[1]
                    self.negetive_instance_list.append(Triple(None, subj, obj))

    def crawl_nege_instances(self):
        if os.path.exists(self.negetive_file_path): return
        self.logger.info("Start collect negetive instances.")
        nege_instance_per_rule = math.ceil(self.positive_num / self.rules_top_k)
        self.rule_list_for_nege = []

        with open(self.filtered_rule_sorted_by_accuracy_file, "r", encoding="UTF-8") as f:
            for idx, line in enumerate(f.readlines()):
                if idx >= self.rules_top_k:
                    break
                rule_chain, accuracy, recall, f1 = line.strip().split("\t")
                self.rule_list_for_nege.append(Rule(rule_chain, float(accuracy), float(recall), float(f1)))

        s_e_list = []
        for idx, one_rule in enumerate(self.rule_list_for_nege):
            self.logger.info("No.{} Rule: {}".format(idx, one_rule))
            triple_pattern = self.utils.rule_parser(one_rule.rule_chain)
            query = "where { " \
                    + triple_pattern \
                    + "FILTER EXISTS {?s " + self.predicate + " ?random}. " \
                    + "FILTER regex(?e,\"^http\"). MINUS {?s " + self.predicate + " ?e}}"
            count_query = "select count(?s) " + query
            entity_query = "select ?s ?e " + query
            s_e_list_temp = self.utils.get_s_e_by_sparql(entity_query, count_query, nege_instance_per_rule * 2)
            if (type(s_e_list_temp) is type([])): s_e_list.extend(s_e_list_temp)
        sample_idx_list = random.sample(range(0, len(s_e_list)), math.ceil(len(s_e_list) / 2))

        with open(self.negetive_file_path, "w", encoding="UTF-8") as f:
            f.write("Num: {}\n".format(len(sample_idx_list)))
            for sample_idx in sample_idx_list:
                f.write("{}\n".format(s_e_list[sample_idx]))

    def generate_one_instance_feature(self, one_instance, rules):
        subj = one_instance.subj
        obj = one_instance.obj
        features = []
        for one_rule in rules:
            raw_rule = one_rule.rule_chain
            triple_pattern = self.utils.rule_parser(raw_rule).replace("?s", subj).replace("?e", obj)
            ask_sparql = "ASK {" + triple_pattern + "}"
            features.append(self.utils.ask_sparql(ask_sparql))
        if np.sum(np.array(features)) == 0:
            self.logger.info("Can't get useful features:\n{}\t{}".format(subj, obj))
            return None
        else:
            return features

    def generate_posi_feature(self):
        if os.path.exists(self.positive_features_path): return
        filtered_rule_list = []
        positive_features = []
        with open(self.filtered_rule_sorted_by_accuracy_file, "r", encoding="UTF-8") as f:
            for idx, line in enumerate(f.readlines()):
                if idx >= self.rules_top_k: break
                rule_chain, precision, recall, f1 = line.strip().split("\t")
                filtered_rule_list.append(Rule(rule_chain, float(precision), float(recall), float(f1)))

        if len(self.positive_instance_list) == 0: self.load_positive_instances()
        for idx, posi_instance in enumerate(self.positive_instance_list):
            self.logger.info(
                "Posi Feature, No.{} Subj {}\t Obj: {}".format(idx, posi_instance.subj, posi_instance.obj))
            one_feature = self.generate_one_instance_feature(posi_instance, filtered_rule_list)
            if one_feature is not None:
                positive_features.append(one_feature)
                self.logger.info("Useful/All: {}/{}".format(len(positive_features), idx + 1))
        np.save(self.positive_features_path, positive_features)
        self.logger.info(positive_features)

    def generate_nege_feature(self):
        if os.path.exists(self.negetive_features_path): return
        filtered_rule_list = []
        negetive_features = []
        with open(self.filtered_rule_sorted_by_accuracy_file, "r", encoding="UTF-8") as f:
            for idx, line in enumerate(f.readlines()):
                if idx >= self.rules_top_k: break
                rule_chain, precision, recall, f1 = line.strip().split("\t")
                filtered_rule_list.append(Rule(rule_chain, float(precision), float(recall), float(f1)))

        if len(self.negetive_instance_list) == 0: self.load_negetive_instances()
        for idx, nege_instance in enumerate(self.negetive_instance_list):
            self.logger.info(
                "Nege Feature, No.{} Subj {}\t Obj: {}".format(idx, nege_instance.subj, nege_instance.obj))
            one_feature = self.generate_one_instance_feature(nege_instance, filtered_rule_list)
            if one_feature is not None:
                negetive_features.append(one_feature)
        np.save(self.negetive_features_path, negetive_features)

    def generate_model(self):
        if os.path.exists(self.model_path):
            self.reload_model()
            return
        epoch = 1000
        mini_batch = 50
        x_posi = np.load(self.positive_features_path)
        y_posi = np.ones(len(x_posi))

        x_nege = np.load(self.negetive_features_path)
        y_nege = np.zeros(len(x_nege))

        lg = LogisticRegression(x_posi.shape[1])

        train_x = np.append(x_posi, x_nege, axis=0)
        train_y = np.append(y_posi, y_nege)

        lg.train(train_x, train_y, epoch, mini_batch)
        lg.saveModel(self.model_path)

    def reload_model(self):
        self.reloaded_model = LogisticRegression(len(self.filtered_rule_sorted_by_accuracy_list))
        self.reloaded_model.loadModel(self.model_path)

    def get_prob(self, one_triple_features):
        train_x = np.array([one_triple_features])
        prob = self.reloaded_model.get_output_prob(train_x)
        return prob

    def test_model(self):
        x_posi = np.load(self.positive_features_path)
        y_posi = np.ones(len(x_posi))

        x_nege = np.load(self.negetive_features_path)
        y_nege = np.zeros(len(x_nege))

        lg = LogisticRegression(x_posi.shape[1])
        lg.loadModel(self.model_path)

        train_x_posi, test_x_posi, train_y_posi, test_y_posi = train_test_split(x_posi, y_posi, train_size=1000)
        train_x_nege, test_x_nege, train_y_nege, test_y_nege = train_test_split(x_nege, y_nege, train_size=1000)
        train_x = np.append(train_x_posi, train_x_nege, axis=0)
        train_y = np.append(train_y_posi, train_y_nege)

        test_x = np.append(test_x_posi, test_x_nege, axis=0)
        test_y = np.append(test_y_posi, test_y_nege)

        lg.test(train_x, train_y)
        lg.test(test_x, test_y)

    def load_rule_sorted_by_recall(self):
        if len(self.filtered_rule_sorted_by_recall_list) != 0: return
        with open(self.filtered_rule_sorted_by_recall_file, "r", encoding="UTF-8") as f:
            for line in f.readlines():
                rule_chain, precision, recall, f1 = line.strip().split("\t")
                self.filtered_rule_sorted_by_recall_list.append(
                    Rule(rule_chain, float(precision), float(recall), float(f1)))

    def get_rule_sorted_by_recall(self):
        if os.path.exists(self.filtered_rule_sorted_by_recall_file): return
        with open(self.filtered_rule_sorted_by_accuracy_file, "r", encoding="UTF-8") as f:
            for idx, line in enumerate(f.readlines()):
                if idx >= self.rules_top_k: break
                rule_chain, precision, recall, f1 = line.strip().split("\t")
                self.filtered_rule_sorted_by_recall_list.append(
                    Rule(rule_chain, float(precision), float(recall), float(f1)))
        self.filtered_rule_sorted_by_recall_list = sorted(self.filtered_rule_sorted_by_recall_list,
                                                          key=lambda k: k.recall, reverse=True)
        with open(self.filtered_rule_sorted_by_recall_file, "w", encoding="UTF-8") as f:
            for one_rule in self.filtered_rule_sorted_by_recall_list:
                f.write("{}\n".format(one_rule))

    def learnRule(self):
        self.logger.info("Start learning rule for {}.".format(self.utils.gen_prefix(self.predicate)))
        self.get_posi_instances()
        self.get_raw_rule()
        self.check_raw_rule()
        self.filtered_sort_checked_rule()
        self.crawl_nege_instances()
        self.generate_posi_feature()
        self.generate_nege_feature()
        self.generate_model()
        self.get_rule_sorted_by_recall()
        self.logger.info("Finish learning rules for {}.".format(self.utils.gen_prefix(self.predicate)))

    def get_rule_sorted_by_recall_by_id(self, idx):
        self.load_rule_sorted_by_recall()
        return self.filtered_rule_sorted_by_recall_list[idx]


if __name__ == "__main__":
    predicate = "<http://dbpedia.org/ontology/birthPlace>"
    rl = RuleLearner(predicate, 2000, 200)
    rl.learnRule()
