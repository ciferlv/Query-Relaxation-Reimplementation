from RuleBased.Classifier import LogisticRegression
from RuleBased.Graph import Graph
import argparse
import numpy as np

from RuleBased.Triple import Rule
from Util import Util
import time

util = Util()
parser = argparse.ArgumentParser()

r_name = "dbo:foundationPlace"
# r_name = "dbo:locationCountry"

r_name_file = "foundationPlace"
# r_name_file = "locationCountry"

# rule_size_to_use = 50
# rule_size_to_use = 100
# rule_size_to_use = 200
# rule_size_to_use = 400
rule_size_to_use = 800
# rule_size_to_use = 1000

# rule model file
parser.add_argument("--rule_file", type=str,
                    default="../../MyData/DBO/United_States/model/{}/rule.txt".format(r_name_file),
                    help="rule file.")

parser.add_argument("--ht_file", type=str, default="./OneCons_eval/{}/ht.txt".format(r_name_file))
parser.add_argument("--rule_model_folder", type=str,
                    default="../../MyData/DBO/United_States/model/{}/{}/".format(r_name_file, rule_size_to_use))
parser.add_argument("--rule_model_file", type=str,
                    default="../../MyData/DBO/United_States/model/{}/{}/pra_model.tar".format(r_name_file,
                                                                                              rule_size_to_use),
                    help="rule model file")
parser.add_argument("--eval_res_file", type=str,
                    default="../../MyData/DBO/United_States/model/{}/{}/eval.txt".format(r_name_file,
                                                                                         rule_size_to_use),
                    help="evaluation result file")

# search scope
parser.add_argument('--search_e2idx_file', type=str, default="../../MyData/DBO/All/entity2id.txt",
                    help="entity idx file for search")
parser.add_argument('--search_r2idx_file', type=str, default="../../MyData/DBO/All/relation2id.txt",
                    help="relation idx file for search")
parser.add_argument('--search_triple2idx_file', type=str, default="../../MyData/DBO/All/triple2id.txt",
                    help="triple idx file for search")

# train scope
parser.add_argument('--train_e2idx_file', type=str, default="../../MyData/DBO/United_States/entity2id.txt",
                    help="entity idx file for train")
parser.add_argument('--train_r2idx_file', type=str, default="../../MyData/DBO/United_States/relation2id.txt",
                    help="relation idx file for train")
parser.add_argument('--train_triple2idx_file', type=str, default="../../MyData/DBO/United_States/triple2id.txt",
                    help="triple idx file for train")
parser.add_argument("--train_feature_file", type=str,
                    default="../../MyData/DBO/United_States/model/{}/pca_train_feature_data.npy".format(r_name_file))
args = parser.parse_args()

def get_graph(e2idx_file, r2idx_file, triple2idx_file):
    graph = Graph()
    graph.e2idx_file = e2idx_file
    graph.r2idx_file = r2idx_file
    graph.triple2idx_file = triple2idx_file
    graph.load_data()
    return graph


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


def load_rule_obj_from_file(my_graph):
    rule_obj_list = []
    for line in open(args.rule_file, 'r', encoding="UTF-8").readlines():
        _, name_key, p, r, f = line.split("\t")
        r_name_rule = name_key.split("=>")
        r_id_rule = [my_graph.r_name2id[tr_name] for tr_name in r_name_rule]

        tmp_rule_obj = Rule(my_graph.r_name2id[r_name])
        tmp_rule_obj.set_r_id_rule(r_id_rule)
        tmp_rule_obj.set_r_name_rule(r_name_rule)
        tmp_rule_obj.set_P_R_F1(p, r, f)

        rule_obj_list.append(tmp_rule_obj)
    return rule_obj_list


def get_rule_model():
    train_graph = get_graph(args.train_e2idx_file, args.train_r2idx_file, args.train_triple2idx_file)
    train_feature_data = np.load(args.train_feature_file)

    rule_obj_list = load_rule_obj_from_file(train_graph)
    feature_size = min([rule_size_to_use, len(rule_obj_list)])
    lg = LogisticRegression(feature_size)

    train_x = train_feature_data[:, 0: feature_size]
    train_y = train_feature_data[:, -1]
    train_y = np.reshape(train_y, [len(train_y), 1])

    new_train_feature_data = np.hstack((train_x, train_y))

    util.createFolder(args.rule_model_folder)

    print("Start training")
    lg.update(new_train_feature_data, epoch=1000, batch_size=128)
    lg.saveModel(args.rule_model_file)


def eval_one_cons():
    print("start evaluate rule size.")
    search_graph = get_graph(args.search_e2idx_file, args.search_r2idx_file, args.search_triple2idx_file)

    rule_obj_list = load_rule_obj_from_file(search_graph)
    rule_obj_list = rule_obj_list[:rule_size_to_use]

    feature_size = min([rule_size_to_use, len(rule_obj_list)])
    lg = LogisticRegression(feature_size)
    lg.loadModel(args.rule_model_file)

    h_id_list, t_id_list = load_ht_from_file(args.ht_file, search_graph)

    print("Cal {}".format(r_name))
    # r_idx = search_graph.r_id_by_name(r_name)

    hits10_r = 0.0
    map_r = 0.0
    mrr_r = 0.0
    time_r = 0
    start_time = time.time()

    for test_example_idx, h_idx in enumerate(h_id_list):
        res_t_prob_list = search_graph.get_tail_from_head_by_rule(h_idx, rule_obj_list, lg)
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

    with open(args.eval_res_file, 'w', encoding="UTF-8") as f:
        f.write("r_name\t{}\n".format(r_name))
        f.write("rule_size_to_use\t{}\n".format(rule_size_to_use))
        f.write("hits10\t{}\n".format(hits10_r))
        f.write("map\t{}\n".format(map_r))
        f.write("mrr\t{}\n".format(mrr_r))
        f.write("time\t{}\n".format(time_r))


if __name__ == "__main__":

    # get_rule_model()
    eval_one_cons()
