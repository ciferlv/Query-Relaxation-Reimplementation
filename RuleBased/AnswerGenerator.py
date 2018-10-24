from RuleBased.RuleLearner import RuleLearner
from RuleBased.SparqlSeg import SparqlSeg
import threading
import numpy as np
from itertools import product
import os

from RuleBased.Util import Util


class AnswerGenerator:
    def __init__(self, sparql_query, question_name):
        self.sparqlSeg = SparqlSeg(sparql_query)
        self.sparqlSeg.analyze_sparql()
        self.pred_rule_dict = {}
        self.positive_num = 2000
        self.rules_top_k = 200
        self.relax_to_top_k = 10
        self.utils = Util()
        self.candidate_entities_set = set()

        self.folder = "./questions/" + question_name + "/"
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        self.entities_file = self.folder + "candidate_entities.txt"

    def get_rule(self):
        thread_list = []
        for triple_pattern in self.sparqlSeg.body_triple_list:
            predicate = triple_pattern.strip().split()[1]
            self.pred_rule_dict[predicate] = RuleLearner(predicate, self.positive_num, self.rules_top_k)
            thread_list.append(threading.Thread(target=self.pred_rule_dict[predicate].learnRule, ))
        [t.start() for t in thread_list]
        [t.join() for t in thread_list]

    def search_candidate_per_body(self, combo, predicate_list):
        replacement = {}
        for idx, predicate in enumerate(predicate_list):
            choosed_predicate_idx = combo[idx]
            if choosed_predicate_idx == 0:
                choosed_predicate = predicate
            else:
                choosed_predicate = self.pred_rule_dict[predicate].get_rule_by_id(choosed_predicate_idx - 1).rule_chain
            replacement[predicate] = choosed_predicate

        rewritted_body_triple_list = []
        for triple in self.sparqlSeg.body_triple_list:
            head, predicate, tail = triple.strip().split()
            replaced_predicate = replacement[predicate]
            if predicate == replaced_predicate:
                rewritted_body_triple_list.append(triple)
            else:
                parsed_rule = self.utils.rule_parser(replaced_predicate)
                parsed_rule = parsed_rule.replace("?s", head).replace("?e", tail.strip("."))
                rewritted_body_triple_list.append(parsed_rule)

        self.utils.get_entity_set_by_sparql(list(self.sparqlSeg.body_vars),
                                            rewritted_body_triple_list, self.candidate_entities_set)

    def search_candidates(self):
        if os.path.exists(self.entities_file):
            return
        self.get_rule()

        thread_list = []

        perm_array = np.arange(0, self.relax_to_top_k)
        predicate_list = list(self.pred_rule_dict.keys())
        for combo in product(perm_array, repeat=len(predicate_list)):
            # self.search_candidate_per_body(combo,predicate_list)
            thread_list.append(threading.Thread(target=self.search_candidate_per_body, args=[combo, predicate_list]))

        [t.start() for t in thread_list]
        [t.join() for t in thread_list]

        if len(self.candidate_entities_set) != 0:
            with open(self.entities_file, "w", encoding="UTF-8") as f:
                for entities in self.candidate_entities_set:
                    f.write(entities + "\n")


if __name__ == "__main__":
    sparql_query = """
        SELECT ?film
        WHERE{
            ?film <http://dbpedia.org/ontology/director> ?p.
            ?film <http://dbpedia.org/ontology/starring> ?p.
            ?p <http://dbpedia.org/ontology/birthPlace> <http://dbpedia.org/resource/North_America>.
        }"""
    ag = AnswerGenerator(sparql_query, "test")
    ag.search_candidates()
