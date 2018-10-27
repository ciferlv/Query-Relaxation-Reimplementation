import threading
import logging

import sys

from RuleBased.ALogger import ALogger
from RuleBased.Util import Util


class Candidate:
    def __init__(self):
        self.util = Util()
        self.logger = ALogger("Candidate.py",True).getLogger()
        self.var_entity_dict = {}
        self.prob_per_triple = {}
        self.path_per_triple = {}
        self.success_rate = 1

    def construct_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.formatter = logging.Formatter("%(filename)s %(funcName)s %(lineno)s - %(message)s",
                                                      "%Y-%m-%d %H:%M:%S")
        if not logger.hasHandlers():
            logger.addHandler(console_handler)
        return logger

    def add_var_entity(self, var, entity):
        assert (var not in self.var_entity_dict)
        self.var_entity_dict[var] = entity

    def set_body_triple_list(self, sparql_body_triple_list):
        self.sparql_body_triple_list = sparql_body_triple_list

    def get_info_per_triple(self, triple, pred_rule_dict):
        features = []
        head, pred, tail = triple.strip().strip(".").split()
        if head.startswith("?"):
            head = self.util.format_uri(self.var_entity_dict[head])
        if tail.startswith("?"):
            tail = self.util.format_uri(self.var_entity_dict[tail])

        ask_query = "ASK {" + head + " " + pred + " " + tail + ".}"
        is_passed = self.util.ask_sparql(ask_query)

        if is_passed == 1:
            self.prob_per_triple[triple] = 1.0
            self.path_per_triple[triple] = [[head + " " + pred + " " + tail]]
            return

        pred_rule = pred_rule_dict[pred]
        pred_rule.load_filtered_rule_sorted_accuracy()
        self.path_per_triple[triple] = []
        for idx, one_rule in enumerate(pred_rule.filtered_rule_sorted_by_accuracy_list):
            if idx >= pred_rule.rules_top_k: break
            parsed_rule = one_rule.parsed_rule.replace("?s", head).replace("?e", tail)
            ask_query = "ASK {" + parsed_rule + "}"
            is_passed = self.util.ask_sparql(ask_query)
            if is_passed == 1:
                self.path_per_triple[triple].append(self.util.get_query_path(parsed_rule))
            features.append(is_passed)
        prob = pred_rule.get_prob(features)
        self.prob_per_triple[triple] = prob

    def get_triples_info(self, pred_rule_dict):
        thread_list = []
        for triple in self.sparql_body_triple_list:
            thread_list.append(threading.Thread(target=self.get_info_per_triple, args=[triple, pred_rule_dict]))
        [t.start() for t in thread_list]
        [t.join() for t in thread_list]
        self.calculate_rate()
        self.logger.info("{} Success Rate: {}".format(self.var_entity2str(), self.success_rate))
        self.logger.info("Success Rate per triple: ")
        self.logger.info("\n{}".format(self.prob_per_triple_str()))
        self.logger.info("\n{}".format(self.path_per_triple_str()))

    def prob_per_triple_str(self):
        res = ""
        for key in self.prob_per_triple:
            res += "{}: {}\n".format(key, self.prob_per_triple[key])
        return res

    def path_per_triple_str(self):
        res = ""
        for key in self.path_per_triple:
            res += key + "\n"
            for path in self.path_per_triple[key]:
                res += "->".join(path) + "\n"
            res += "\n"
        return res

    def calculate_rate(self):
        for key in self.prob_per_triple.keys():
            self.success_rate *= self.prob_per_triple[key]

    def var_entity2str(self):
        res = ""
        for key in self.var_entity_dict:
            res += key + ";" + self.var_entity_dict[key] + "\t"
        return res.strip()
