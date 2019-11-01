import argparse
import time

from Empty_Answer_Query import eaqs
from RuleBased.Graph import Graph
from RuleBased.SparqlParser import SparqlParser
from RuleBased.Params import pca_or_cwa
from Util import Util


def load_ht_from_file(file_path, graph):
    h_idx_list = []
    t_idx_list = []
    with open(file_path, 'r', encoding="UTF-8") as f:
        for idx, line in enumerate(f.readlines()):
            if idx == 0:
                continue

            e_idx_list = [graph.get_e_idx_by_e_name(e_name) for e_name in line.strip().split("\t")]
            h_idx_list.append(e_idx_list[0])
            t_idx_list.append(e_idx_list[1:])
    return h_idx_list, t_idx_list


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

final_map_r = 0.0
final_hits10_r = 0.0
final_mrr_r = 0.0
final_time = 0

all_metric_file = "./OneCons_eval/pra_all_metric_normal.txt"

for r_name in r_name_list:
    graph.rule_file = "../../MyData/DBO/United_States/model/{}/rule.txt".format(r_name.split(":")[-1])
    graph.model_file = "../../MyData/DBO/United_States/model/{}/{}_model.tar".format(r_name.split(":")[-1],
                                                                                     pca_or_cwa)
    graph.rule_num_to_use_file = "../../MyData/DBO/United_States/model/{}/rule_num_to_use.txt".format(
        r_name.split(":")[-1])

    res_folder = "./OneCons_eval/{}/".format(r_name.split(":")[-1])
    util.createFolder(res_folder)
    metric_file = "{}{}_rdqe_metric.txt".format(res_folder, pca_or_cwa)
    rule_obj_list = graph.load_rule_obj_from_file(r_name)
    rule_model = graph.get_pra_model4r(graph.r_id_by_name(r_name))

    ht_file = res_folder + "ht.txt"
    h_id_list, t_id_list = load_ht_from_file(ht_file, graph)

    print("Cal {}".format(r_name))
    r_idx = graph.r_id_by_name(r_name)

    hits10_r = 0.0
    map_r = 0.0
    mrr_r = 0.0
    time_r = 0
    start_time = time.time()

    for test_example_idx, h_idx in enumerate(h_id_list):
        res_t_prob_list = graph.get_tail_from_head_by_rule(h_idx, rule_obj_list, rule_model)
        t_start_time = time.time()
        cnt = 0
        rr = 0.0
        ap = 0.0
        hits10 = 0
        for res_ht_idx, t_prob in enumerate(res_t_prob_list):
            t_idx = t_prob[0]
            if t_idx in t_id_list[test_example_idx]:
                if rr == 0:
                    rr = 1.0 / (1 + res_ht_idx)
                if res_ht_idx < 10:
                    hits10 = 1
                cnt += 1
                ap += cnt / (res_ht_idx + 1)

        if cnt != 0:
            ap /= cnt
        map_r += ap
        mrr_r += rr
        hits10_r += hits10
        t_end_time = time.time()
        time_r -= t_end_time - t_start_time
    end_time = time.time()
    time_r = time_r + (end_time - start_time)

    mrr_r = mrr_r / len(h_id_list)
    hits10_r /= len(h_id_list)
    map_r /= len(h_id_list)
    time_r /= len(h_id_list)

    print("mrr:{}\thits10:{}\tmap:{}\ttime:{}".format(mrr_r, hits10_r, map_r, time_r))

    final_map_r += map_r
    final_hits10_r += hits10_r
    final_mrr_r += mrr_r
    final_time += time_r
    with open(metric_file, 'w', encoding="UTF-8") as f:
        f.write("hits10\t{}\n".format(hits10_r))
        f.write("map\t{}\n".format(map_r))
        f.write("mrr\t{}\n".format(mrr_r))
        f.write("time\t{}\n".format(time_r))

with open(all_metric_file, 'w', encoding="UTF-8") as f:
    f.write("hits10\t{}\n".format(final_hits10_r / len(r_name_list)))
    f.write("map\t{}\n".format(final_map_r / len(r_name_list)))
    f.write("mrr\t{}\n".format(final_mrr_r / len(r_name_list)))
    f.write("avg\t{}\n".format(final_time / len(r_name_list)))
