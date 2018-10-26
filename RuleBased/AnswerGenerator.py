from RuleBased.RuleLearner import RuleLearner
from RuleBased.SparqlSeg import SparqlSeg
import threading
import numpy as np
from itertools import product
import os
import json

from RuleBased.Util import Util
from RuleBased.unit.Candidate import Candidate


class AnswerGenerator:
    def __init__(self, sparql_query, question_name):
        self.sparqlSeg = SparqlSeg(sparql_query)
        self.sparqlSeg.analyze_sparql()
        self.pred_rule_dict = {}
        self.positive_num = 2000
        self.rules_top_k = 200
        self.relax_to_top_k = 10
        self.utils = Util()
        self.cand_list = []

        self.folder = "./questions/" + question_name + "/"
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        self.cand_entities_file = self.folder + "candidate_entities.txt"
        self.display_cand_file = self.folder + "cands_display.json"

    def load_cands(self):
        if len(self.cand_list) == 0:
            with open(self.cand_entities_file,"r",encoding="UTF-8") as f:
                for line in f.readlines():
                    temp_cand = Candidate()
                    for var_entity in line.strip().split("\t"):
                        var,entity = var_entity.strip().split(":")
                        temp_cand.add_var_entity(var=var,entity=entity)
                    self.cand_list.append(temp_cand)
        for cand in self.cand_list:
            cand.set_body_triple_list(self.sparqlSeg.body_triple_list)

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
                choosed_predicate = self.pred_rule_dict[predicate].get_rule_sorted_by_recall_by_id(choosed_predicate_idx - 1).rule_chain
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

        temp_cand_list = self.utils.get_entity_set_by_sparql(list(self.sparqlSeg.body_vars),
                                            rewritted_body_triple_list)
        for one_cand in temp_cand_list:
            cand = Candidate()
            for var_entity in one_cand:
                cand.add_var_entity(var_entity[0],var_entity[1])
            self.cand_list.append(cand)

    def search_candidates(self):
        if os.path.exists(self.cand_entities_file):
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

        if len(self.cand_list) != 0:
            with open(self.cand_entities_file, "w", encoding="UTF-8") as f:
                for cand in self.cand_list:
                    f.write(cand.var_entity2str() + "\n")

    def display_cands(self):
        self.load_cands()
        for cand in self.cand_list:
            cand.get_triples_info()
        with open(self.display_cand_file,"w",encoding="UTF-8") as f:
            f.write(json.dump(self.cand_list))




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
