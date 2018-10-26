import threading

from RuleBased.Util import Util


class Candidate:
    def __init__(self):
        self.util = Util()
        self.var_entity_dict = {}
        self.correct_prob_per_triple = {}
        self.established_path_per_triple = {}
        self.success_rate = 1

    def add_var_entity(self, var, entity):
        assert (var not in self.var_entity_dict)
        self.var_entity_dict[var] = entity

    def set_body_triple_list(self, sparql_body_triple_list):
        self.sparql_body_triple_list = sparql_body_triple_list

    def get_info_per_triple(self, triple, pred_rule_dict):
        features = []
        head, pred, tail = triple.strip().split()
        if head.startswith("?"):
            head = self.var_entity_dict[head]
        if tail.startswith("?"):
            tail = self.var_entity_dict[tail.strip(".")]

        pred_rule = pred_rule_dict[pred]
        pred_rule.load_filtered_rule_sorted_accuracy()
        self.established_path_per_triple[triple] = []
        for idx, one_rule in enumerate(pred_rule.filtered_rule_sorted_by_accuracy_list):
            if idx >= pred_rule.rules_top_k: break
            parsed_rule = one_rule.parsed_rule.replace("?s", head).replace("?e", tail.strip("."))
            ask_query = "ASK {" + parsed_rule + "}"
            is_passed = self.util.ask_sparql(ask_query)
            if is_passed == 1:
                self.established_path_per_triple[triple].append(self.util.get_query_path(parsed_rule))
            features.append(is_passed)
        prob = pred_rule.get_prob(features)
        self.correct_prob_per_triple[triple] = prob

    def get_triples_info(self, pred_rule_dict):
        thread_list = []
        for triple in self.sparql_body_triple_list:
            thread_list.append(threading.Thread(target=self.get_info_per_triple, args=[triple, pred_rule_dict]))
        [t.start() for t in thread_list]
        [t.join() for t in thread_list]

    def calculate_rate(self):
        for key in self.correct_prob_per_triple.keys():
            self.success_rate *= self.correct_prob_per_triple[key]


    def var_entity2str(self):
        res = ""
        for key in self.var_entity_dict:
            res += key + ":" + self.var_entity_dict[key] + "\t"
        return res.strip()
