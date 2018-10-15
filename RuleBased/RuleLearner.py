from SPARQLWrapper import SPARQLWrapper, JSON
import random
import math
import os
import queue

from RuleBased.Util import Util

sparql_database = "http://210.28.132.61:8898/sparql"


class Triple:
    def __init__(self, searched_dict=None, subj=None, obj=None):
        self.sparql = SPARQLWrapper(sparql_database)
        if searched_dict is None:
            self.subj = subj
            self.obj = obj
        else:
            self.subj = searched_dict['s']['value']
            self.obj = searched_dict['o']['value']
        self.rule_set = set()

    def get_pred(self, sparql, position):
        self.sparql.setQuery(sparql)
        self.sparql.setReturnFormat(JSON)
        results = self.sparql.query().convert()
        for res in results['results']['bindings']:
            one_rule = ""
            for i, key in enumerate(res.keys()):
                one_rule += position[i] + res[key]['value'] + ";"
            self.rule_set.add(one_rule)

    def display_rule(self):
        for one_rule in self.rule_set:
            print(one_rule)

    def search_rule(self):
        query_filter = "FILTER regex(?o, \"^http\"). FILTER (?o != " + self.subj + "). FILTER (?o != " + self.obj + ").}"
        query0 = "select ?p1 where { " + self.subj + " ?p1 " + self.obj + "}"
        query1 = "select ?p1 where { " + self.obj + " ?p1 " + self.subj + "}"
        query2 = "select ?p1 ?p2 where{ " + self.subj + " ?p1 ?o.\n?o ?p2 " + self.obj + "." + query_filter
        query3 = "select ?p1 ?p2 where{ " + self.subj + " ?p1 ?o.\n" + self.obj + " ?p2 ?o." + query_filter
        query4 = "select ?p1 ?p2 where{ ?o ?p1 " + self.subj + ".\n" + self.obj + " ?p2 ?o." + query_filter
        query5 = "select ?p1 ?p2 where{ ?o ?p1 " + self.subj + ".\n ?o ?p2" + self.obj + "." + query_filter

        self.get_pred(query0, ['+'])
        self.get_pred(query1, ['-'])
        self.get_pred(query2, ['+', '+'])
        self.get_pred(query3, ['+', '-'])
        self.get_pred(query4, ['-', '-'])
        self.get_pred(query5, ['-', '+'])

    def __str__(self):
        msg = self.subj + "\t" + self.obj
        return msg


class Rule:
    def __init__(self, rule_chain, accuracy, recall, f1):
        self.rule_chain = rule_chain
        self.accuracy = accuracy
        self.recall = recall
        self.f1 = f1

    def calculate_f1(self):
        self.f1 = self.accuracy * self.recall * 2 / (self.accuracy + self.recall)

    def __lt__(self, other):
        # return self.f1 > other.f1
        return self.accuracy > other.accuracy

    def __gt__(self, other):
        # return self.f1 < other.f1
        return self.accuracy < other.accuracy

    def __eq__(self, other):
        # return self.f1 == other.f1
        return self.accuracy == other.accuracy

    def __str__(self):
        msg = "{}\t{}\t{}\t{}".format(self.rule_chain, self.accuracy, self.recall, self.f1)
        return msg


class RuleLearner:
    def __init__(self, predicate, positive_num):
        self.utils = Util()
        self.predicate = predicate
        self.sparql = SPARQLWrapper(sparql_database)
        self.positive_instances = []
        self.rule_list = []
        self.positive_num = positive_num
        self.folder = "./data/" + predicate.split("/")[-1][:-1] + "/"
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        self.positive_file_path = self.folder + "positive_instances.txt"
        self.negetive_file_path = self.folder + "negetive_instances.txt"
        self.checked_rule_file = self.folder + "checked_rule.txt"
        self.filtered_sorted_rule_file = self.folder + "filtered_sorted_rule.txt"

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
        with open(self.positive_file_path, "r", encoding="UTF-8") as f:
            for idx, line in enumerate(f.readlines()):
                if idx == 0:
                    continue
                else:
                    subj = line.strip().split()[0]
                    obj = line.strip().split()[1]
                    self.positive_instances.append(Triple(None, subj, obj))

    def crawl_positive_instances(self):
        triple_pattern = "?s " + self.predicate + " ?o"
        res_num = self.utils.get_num_by_sparql("""select count(?s) where {""" + triple_pattern + """}""")
        search_times, num_per_time = self.calculate_search_num_per_time(total_num=res_num,
                                                                        positive_num=self.positive_num)

        print("Search_times: {}.".format(search_times))
        print("Num_per_time: {}.".format(num_per_time))

        for i in range(search_times):
            sparql_query = "select ?s ?o where {" + triple_pattern + "} limit 10000  offset " + str(10000 * i)
            print(sparql_query)
            self.sparql.setQuery(sparql_query)
            self.sparql.setReturnFormat(JSON)
            results = self.sparql.query().convert()
            res_list = results['results']['bindings']
            for sampled_num in random.sample(range(0, 10000), num_per_time):
                self.positive_instances.append(Triple(res_list[sampled_num]))
        with open(self.positive_file_path, "w") as f:
            f.write("Num: {}".format(len(self.positive_instances)))
            for instance in self.positive_instances:
                f.write("{}\n".format(instance))

    def get_posi_instances(self):
        if os.path.exists(self.positive_file_path):
            self.load_positive_instances()
        else:
            self.crawl_positive_instances()

    def get_rule(self):
        self.raw_rule_file = self.folder + "raw_rule.txt"
        if os.path.exists(self.raw_rule_file):
            with open(self.raw_rule_file, "r") as f:
                for line in f.readlines():
                    self.rule_list.append(line)
        else:
            for i, instance in enumerate(self.positive_instances):
                print("{}".format(i))
                instance.search_rule()
                self.rule_list.extend(list(instance.rule_set))
            self.rule_list = list(set(self.rule_list))
            with open(self.raw_rule_file, "w") as f:
                for rule in self.rule_list:
                    f.write(rule + "\n")

    def rule_parser(self, raw_rule):
        prev = "?s"
        triple_pattern = ""
        raw_rule_array = raw_rule.strip().strip(";").split(";")
        for idx, pred in enumerate(raw_rule_array):

            if idx == len(raw_rule_array) - 1:
                next_singal = "?e"
            else:
                next_singal = "?o" + str(idx)

            if pred[0] is "+":
                triple_pattern += prev + " <" + pred[1:] + "> " + next_singal + ".\n"
            else:
                triple_pattern += next_singal + " <" + pred[1:] + "> " + prev + ".\n"
            prev = next_singal
        return triple_pattern

    def rule_checker(self):
        self.checked_rule_dict = {}
        if not os.path.exists(self.checked_rule_file):
            total_num_query = "select count(?s) where {?s " + self.predicate + " ?e.}"
            total_num = self.utils.get_num_by_sparql(total_num_query)
            for idx, raw_rule in enumerate(self.rule_list):

                print("No.{} {}".format(idx, raw_rule.strip()))
                if raw_rule.strip().split(";")[0][1:] == self.predicate[1:-1]:
                    continue

                triple_pattern = self.rule_parser(raw_rule)
                query1 = "select count(?s) where{ " + triple_pattern + "FILTER EXISTS { ?s " + self.predicate + " ?random.}}"
                query2 = "select count(?s)where{ " + triple_pattern + "?s " + self.predicate + " ?e.}"

                correct_num = self.utils.get_num_by_sparql(query2)
                retrived_num = self.utils.get_num_by_sparql(query1)

                accuracy = correct_num * 1.0 / retrived_num
                recall = correct_num / total_num
                f1 = 2 * accuracy * recall / (accuracy + recall)

                self.checked_rule_dict[raw_rule] = [accuracy, recall, f1]

                print("Accuracy: {}\tRecall: {}\tF1: {}".format(accuracy, recall, f1))

            with open(self.checked_rule_file, "w") as f:
                for key in self.checked_rule_dict:
                    rule_metric_list = self.checked_rule_dict[key]
                    f.write("{}\t{}\t{}\t{}\n".format(key.strip(), rule_metric_list[0],
                                                      rule_metric_list[1], rule_metric_list[2]))

    def rule_filter_and_sorter(self):
        self.filtered_rule_heap = queue.PriorityQueue()
        if not os.path.exists(self.filtered_sorted_rule_file):
            with open(self.checked_rule_file, "r") as f:
                for line in f.readlines():
                    accuracy = float(line.split("\t")[1])
                    recall = float(line.split("\t")[2])
                    f1 = float(line.split("\t")[3])
                    if accuracy < 0.001 or recall < 0.001:
                        continue
                    self.filtered_rule_heap.put(
                        Rule(line.split("\t")[0], accuracy, recall, f1))
            with open(self.filtered_sorted_rule_file, "w") as f:
                while not self.filtered_rule_heap.empty():
                    f.write("{}\n".format(self.filtered_rule_heap.get()))

    def get_nege_instances(self):
        self.rule_list = []
        with open(self.filtered_sorted_rule_file, "r", encoding="UTF-8") as f:
            for line in f.readlines():
                rule_chain = line.split("\t")[0]
                accuracy = float(line.split("\t")[1])
                recall = float(line.split("\t")[2])
                f1 = float(line.split("\t")[3])
                self.rule_list.append(Rule(rule_chain, accuracy, recall, f1))
        for one_rule in self.rule_list:
            triple_pattern = self.rule_parser(one_rule.rule_chain)
            query = "select ?s ?e where { " \
                    + triple_pattern \
                    + "FILTER EXISTS {?s " + self.predicate + " ?random}. " \
                    + "FILTER regex(?e,\"^http\"). MINUS {?s " + self.predicate + " ?e}}"
            print(query)


def learnRule(predicate):
    rl = RuleLearner(predicate, 2000)
    rl.get_posi_instances()
    rl.get_rule()
    rl.rule_checker()
    rl.rule_filter_and_sorter()
    rl.get_nege_instances()


if __name__ == "__main__":
    predicate = "<http://dbpedia.org/ontology/birthPlace>"
    learnRule(predicate)
