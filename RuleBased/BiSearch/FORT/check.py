import os
import threading

from RuleBased.BiSearch.Graph import Graph
from RuleBased.BiSearch.SparqlParser import SparqlParser
from RuleBased.Params import file_path_seg, max_step, rule_num4train
from RuleBased.Util import Util

# relation_list = ['<http://dbpedia.org/ontology/director>',
#                  '<http://dbpedia.org/ontology/starring>',
#                  '<http://dbpedia.org/ontology/birthPlace>']

util = Util()
# source_folder = "./source/"
source_folder = "F:\\Data\\FB15K-237\\source\\"
output_folder = "F:\\Data\\FB15K-237\\output\\"
e2idx_file = source_folder + "e2idx.txt"
r2idx_file = source_folder + "r2idx.txt"
triple2idx_file = source_folder + "triple2idx.txt"


def test_modules_train():
    # r_name_list = ['/film/film/language', '/location/location/contains', '/people/person/place_of_birth',
    #                '/film/actor/film./film/performance/film']

    r_name_list = ['/music/genre/artists']

    for r_name in r_name_list:
        folder = output_folder + util.gen_prefix(r_name) + file_path_seg
        if not os.path.isdir(folder): os.makedirs(folder)

    graph = Graph(e2idx_file, r2idx_file, triple2idx_file)
    graph.load_data()
    r_idx_list = [graph.r2idx[relation] for relation in r_name_list]

    # thread_list = []
    for idx, r_idx in enumerate(r_idx_list):
        folder = output_folder + util.gen_prefix(r_name_list[idx]) + file_path_seg
        # thread_list.append(
        #     threading.Thread(target=graph.get_r_model, args=[r_idx, max_step, top_rules_to_use_num, folder]))
        # graph.get_r_model(r_idx=r_idx, folder=folder)
        graph.test_model(graph.idx2r[r_idx], folder=folder)
    #
    # [t.start() for t in thread_list]
    # [t.join() for t in thread_list]


def test_sparql_parser():
    sparql = """
    select * where {
        
        /m/06dv3	 /film/actor/film./film/performance/film	?p.
    }
    """

    sp = SparqlParser(sparql)
    sp.parse_sparql()

    r_name_list = ['/film/film/language', '/film/actor/film./film/performance/film']

    graph = Graph(e2idx_file, r2idx_file, triple2idx_file)
    graph.load_data()
    r_idx_list = [graph.r2idx[relation] for relation in r_name_list]

    # list, [Rule(),Rule(),Rule(),...]
    r_rules_dict = {}
    for r_idx in r_idx_list:
        r_rules_dict[r_idx] = graph.get_top_k_rules(r_idx, 5, 'P')

    sp.execute_var1BGP(r_rules_dict, graph)
    sp.execute_var2BGP(r_rules_dict, graph)


if __name__ == "__main__":
    # test_modules_train()
    test_sparql_parser()
