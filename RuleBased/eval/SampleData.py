import random

from Empty_Answer_Query import eaqs
from RuleBased.Graph import Graph
from RuleBased.Params import pca_or_cwa
from RuleBased.SparqlParser import SparqlParser
from Util import Util

util = Util()

search_folder = "../../MyData/DBO/All/"

graph = Graph()
graph.e2idx_file = search_folder + "entity2id.txt"
graph.r2idx_file = search_folder + "relation2id.txt"
graph.triple2idx_file = search_folder + "triple2id.txt"

graph.load_data()

r_name_list = []
r_idx_list = []

for sparql in eaqs:
    sp = SparqlParser(sparql=sparql)
    sp.parse_sparql()
    for relation in sp.r_name_list:
        r_name_list.append(relation)
    r_name_list = list(set(r_name_list))
r_idx_list = [graph.r_id_by_name(r_name) for r_name in r_name_list]

for r_name in r_name_list:
    res_folder = "./OneCons_eval/{}/".format(r_name.split(":")[-1])
    util.createFolder(res_folder)
    res_file = "{}ht.txt".format(res_folder)

    r_id = graph.r_id_by_name(r_name)
    ht_id_list = random.sample(graph.r2ht[r_id], 1000)
    h_id_list = [ht_id[0] for ht_id in ht_id_list]
    t_id_list_list = []
    for h_id in h_id_list:
        t_id_list_list.append(graph.node_dict[h_id].get_r_value(r_id))

    with open(res_file, 'w', encoding="UTF-8") as f:
        f.write("{}\n".format(r_name))
        for id, h_id in enumerate(h_id_list):
            f.write("{}\t".format(graph.get_e_name_by_e_idx(h_id)))
            for t_id in t_id_list_list[id]:
                f.write("{}\t".format(graph.get_e_name_by_e_idx(t_id)))
            f.write("\n")
