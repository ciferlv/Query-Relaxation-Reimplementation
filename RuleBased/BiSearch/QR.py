import os
import threading
import numpy as np
import queue

from RuleBased.BiSearch.Graph import Graph
from RuleBased.BiSearch.SparqlParser import SparqlParser
from RuleBased.Params import file_path_seg, ht_conn, max_step
from RuleBased.Util import Util

util = Util()
source_folder = "./source/"
source_folder = "F:\\Data\\FB15K-237\\source\\"
output_folder = "F:\\Data\\FB15K-237\\output\\"
e2idx_file = source_folder + "e2idx.txt"
r2idx_file = source_folder + "r2idx.txt"
triple2idx_file = source_folder + "triple2idx.txt"

sparql = """
    SELECT ?film WHERE{
        ?film <http://dbpedia.org/ontology/director> ?p.
        ?film <http://dbpedia.org/ontology/starring> ?p.
        ?p <http://dbpedia.org/ontology/birthPlace> <http://dbpedia.org/resource/North_America>.
    }
    """

graph = Graph(e2idx_file, r2idx_file, triple2idx_file)
graph.load_data()

sp = SparqlParser(sparql=sparql)
sp.parse_sparql()

r_name_list = sp.r_name_list
r_idx_list = [graph.r2idx[relation] for relation in r_name_list]
r_rules_dict = {}


def train_rules():
    thread_list = []
    for idx, r_idx in enumerate(r_idx_list):
        folder = output_folder + util.gen_prefix(r_name_list[idx]) + file_path_seg
        if not os.path.isdir(folder):
            os.makedirs(folder)
        thread_list.append(threading.Thread(target=graph.get_r_model, args=[r_idx, folder]))
        # graph.get_r_model(r_idx=r_idx, max_step=3, top_rules_num=200, folder=folder)
    [t.start() for t in thread_list]
    [t.join() for t in thread_list]


def gen_rules_dict():
    for r_idx in r_idx_list:
        r_rules_dict[r_idx] = graph.get_top_k_rules(r_idx, 5, 'P')

