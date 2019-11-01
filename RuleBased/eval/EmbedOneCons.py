from Empty_Answer_Query import eaqs
from RuleBased.Graph import Graph
from RuleBased.SparqlParser import SparqlParser
from Util import Util
from RuleBased.Params import transe_embed

util = Util()

print("Start")


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


search_folder = "../../MyData/EmbedDBO/"

graph = Graph()
graph.e2idx_file = search_folder + "entity2id.txt"
graph.r2idx_file = search_folder + "relation2id.txt"
graph.triple2idx_file = search_folder + "triple2id.txt"

if transe_embed is True:
    transe_embed = False

graph.load_data()
graph.load_er_embedding()

r_name_list = []

for sparql in eaqs:
    sp = SparqlParser(sparql=sparql)
    sp.parse_sparql()
    for relation in sp.r_name_list:
        r_name_list.append(relation)
    r_name_list = list(set(r_name_list))

final_map = 0.0
final_hits10 = 0.0
final_mrr = 0.0
final_time = 0

all_metric_file = "./OneCons_eval/embed_all_metric_normal.txt"

print("Start")
for r_name in r_name_list:
    print("Cal {}".format(r_name))
    res_folder = "./OneCons_eval/{}/".format(r_name.split(":")[-1])
    util.createFolder(res_folder)
    metric_file = "{}embed_metric.txt".format(res_folder)

    ht_file = res_folder + "ht.txt"
    h_id_list, t_id_list = load_ht_from_file(ht_file, graph)

    r_id = graph.r_id_by_name(r_name)
    r_id_list = [r_id] * len(h_id_list)

    map_r, mrr_r, hits10_r, mean_rank_r, avg_time = graph.ekg.get_map_mrr_hits10_meanRank_avg_time_embed_1cons(
        h_id_list,
        r_id_list,
        t_id_list)

    final_map += map_r
    final_hits10 += hits10_r
    final_mrr += mrr_r
    final_time += avg_time

    print("map:{}\tmrr:{}\thits10:{}\tavg time:{}".format(map_r, mrr_r, hits10_r, avg_time))

    with open(metric_file, 'w', encoding="UTF-8") as f:
        f.write("hits10\t{}\n".format(hits10_r))
        f.write("map\t{}\n".format(map_r))
        f.write("mrr\t{}\n".format(mrr_r))
        f.write("avg time:\t{}\n".format(avg_time))

with open(all_metric_file, 'w', encoding="UTF-8") as f:
    f.write("hits10\t{}\n".format(final_hits10 / len(r_name_list)))
    f.write("map\t{}\n".format(final_map / len(r_name_list)))
    f.write("mrr\t{}\n".format(final_mrr / len(r_name_list)))
    f.write("avg\t{}\n".format(final_time / len(r_name_list)))
