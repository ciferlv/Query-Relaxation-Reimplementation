import os
import threading
import numpy as np
import queue

from RuleBased.BiSearch.Graph import Graph
from RuleBased.BiSearch.MyThread import MyThread
from RuleBased.BiSearch.SparqlParser import SparqlParser
from RuleBased.Params import file_path_seg, ht_conn, max_step
from RuleBased.Util import Util

util = Util()
# source_folder = "F:\\Data\\FB15K-237\\source\\"
# output_folder = "F:\\Data\\FB15K-237\\output\\"
# e2idx_file = source_folder + "e2idx.txt"
# r2idx_file = source_folder + "r2idx.txt"
# triple2idx_file = source_folder + "triple2idx.txt"

source_folder = "F:\\Data\\dbpedia\\"
model_folder = "F:\\Data\\dbpedia\\model\\"
e2idx_file = source_folder + "e2idx_shortcut.txt"
r2idx_file = source_folder + "r2idx_shortcut.txt"
triple2idx_file = source_folder + "triple2idx.txt"

sparql = """
    SELECT ?film WHERE{
        ?film dbo:director ?p.
        ?film dbo:starring ?p.
        ?p dbo:birthPlace dbr:North_America.
    }
    """

graph = Graph(e2idx_file, r2idx_file, triple2idx_file)
graph.load_data()

sp = SparqlParser(sparql=sparql)
sp.parse_sparql()

r_name_list = sp.r_name_list
r_idx_list = [graph.r2idx[relation] for relation in r_name_list]
r_rules_dict = {}
r_model_dict = {}


def train_rules():
    print("Start Training Rules.")
    thread_list = []
    for idx, r_idx in enumerate(r_idx_list):
        folder = model_folder + util.gen_prefix(r_name_list[idx]) + file_path_seg
        if not os.path.isdir(folder):
            os.makedirs(folder)
        thread_list.append(MyThread(graph.get_r_model, [r_idx, folder]))
        # graph.get_r_model(r_idx=r_idx, folder=folder)
    [t.start() for t in thread_list]
    [t.join() for t in thread_list]

    for idx, r_idx in enumerate(r_idx_list):
        r_model_dict[r_idx] = thread_list[idx].get_result()


def gen_rules_dict():
    print("Start collect top rules by Precision.")
    for r_idx in r_idx_list:
        r_rules_dict[r_idx] = graph.get_top_k_rules(r_idx, 5, 'P')


def get_candidates():
    print("Start generating candidates.")

    print("Start executing 1 var BGP.")
    sp.execute_var1BGP(r_rules_dict, graph)

    print("Start executin 2 var BGP.")
    sp.execute_var2BGP(r_rules_dict, graph)

    print("Print cands.")
    [print(var + "\t", end="") for var in sp.var_list]
    print()
    for cand in sp.res:
        print("{}\n".format("\t".join(cand)))


if __name__ == "__main__":
    train_rules()
    gen_rules_dict()
    get_candidates()
