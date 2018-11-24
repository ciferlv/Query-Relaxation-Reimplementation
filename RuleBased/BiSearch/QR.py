import os
import time

from RuleBased.BiSearch.Graph import Graph
from RuleBased.BiSearch.SparqlParser import SparqlParser
from RuleBased.Params import file_path_seg, rules_num_2_search_cands
from RuleBased.VirtuosoSearch.Util import Util

util = Util()
# source_folder = "F:\\Data\\FB15K-237\\source\\"
# output_folder = "F:\\Data\\FB15K-237\\output\\"
# e2idx_file = source_folder + "e2idx.txt"
# r2idx_file = source_folder + "r2idx.txt"
# triple2idx_file = source_folder + "triple2idx.txt"

model_scope = "Japan"
# model_scope = "Asia"
search_scope = "United_States"
# search_scope = "All"
source_folder = "F:\\Data\\dbpedia\\" + search_scope + "\\"
model_folder = "F:\\Data\\dbpedia\\" + model_scope + "\\model\\"
e2idx_file = source_folder + "e2idx_shortcut.txt"
r2idx_file = source_folder + "r2idx_shortcut.txt"
triple2idx_file = source_folder + "triple2idx.txt"

# sparql = """
#     SELECT ?film WHERE{
#         ?film dbo:director ?p.
#         ?film dbo:starring ?p.
#         ?p dbo:birthPlace dbr:North_America.
#     }
#     """

sparql = """
    SELECT ?film WHERE{
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
'''
Dict for r_idx and its list of rule object.
{
    r_idx:[Rule(),Rule(),...],
    r_idx:[Rule(),Rule(),...],
    ...
}
'''
r_rules_dict = {}
'''
Dict for r_idx and its trained model.
{
    r_idx: LogisticRegression Model
    r_idx: LogisticRegression Model
    ...
}
'''
r_model_dict = {}


def train_rules():
    start_time = time.time()
    print("Start Training Rules.")
    for idx, r_idx in enumerate(r_idx_list):
        print("Going to train R:{}".format(graph.idx2r[r_idx]))
        folder = model_folder + util.gen_prefix(r_name_list[idx]).replace(':', '_') + file_path_seg
        if not os.path.isdir(folder):
            os.makedirs(folder)
        model = graph.get_pra_model4r(r_idx=r_idx, folder=folder)
        r_model_dict[r_idx] = model
    end_time = time.time()
    print("Finish train rules and get the model. Elapsed: {}.".format(end_time - start_time))


def test_rules():
    source_folder = "F:\\Data\\dbpedia\\Canada\\"
    e2idx_file = source_folder + "e2idx_shortcut.txt"
    r2idx_file = source_folder + "r2idx_shortcut.txt"
    triple2idx_file = source_folder + "triple2idx.txt"

    model_folder = source_folder + "test_model\\"
    if not os.path.isdir(model_folder):
        os.makedirs(model_folder)

    graph = Graph(e2idx_file, r2idx_file, triple2idx_file)
    graph.load_data()

    print("Start Testing Rules.")
    for idx, r_idx in enumerate(r_idx_list):
        metric_record_file = model_folder + graph.get_localname(graph.get_r_name_by_r_idx(r_idx)) + "_metric.txt"
        print("Going to train R:{}".format(graph.idx2r[r_idx]))
        model = r_model_dict[r_idx]
        rule_list = graph.get_rule4train_from_mysql(r_idx)
        graph.test_model(r_idx, model, rule_list, metric_record_file)


def gen_rules_dict():
    print("Start collect top rules by Precision.")
    for r_idx in r_idx_list:
        r_rules_dict[r_idx] = graph.get_top_k_rules(r_idx, rules_num_2_search_cands, 'P')


def get_candidates():
    start_time = time.time()
    print("Start generating candidates.")

    print("Start executing 1 var BGP.")
    sp.execute_var1BGP(r_rules_dict, graph)

    print("Start executin 2 var BGP.")
    sp.execute_var2BGP(r_rules_dict, graph)

    print("Start normalize searched res.")
    sp.normalize_searched_res()

    print("Display result.")
    sp.display_searched_res(graph)

    print("Calculate confidence for candidates.")
    sp.gen_conf_and_rule_path(r_rules_dict, r_model_dict, graph)

    print("Print cands.")
    sp.display_cands(graph)
    end_time = time.time()
    print("Finishing generating and displaying candidates. Epalsed: {}.".format(end_time - start_time))


if __name__ == "__main__":
    # train_rules()
    # test_rules()
    # gen_rules_dict()
    get_candidates()
