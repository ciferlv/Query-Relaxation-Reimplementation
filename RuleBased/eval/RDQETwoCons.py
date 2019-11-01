from ALogger import ALogger
from RuleBased.Graph import Graph
from RuleBased.Params import pca_or_cwa
from Util import Util
import time


class TwoConts:
    def __init__(self):

        self.r_rules_dict = {}
        self.r_model_dict = {}

        self.util = Util()
        self.logger = ALogger("Graph", True).getLogger()

    def _setGraph(self, graph):
        self.graph = graph

    def _set_r_name(self, r_name1, r1_direction, r_name2, r2_direction):
        self.r_name1 = r_name1
        self.r_name2 = r_name2
        self.r1_direction = r1_direction
        self.r2_direction = r2_direction

        self.r1_idx = self.graph.r_id_by_name(self.r_name1)
        self.r2_idx = self.graph.r_id_by_name(self.r_name2)
        self.r_idx_list = [self.r1_idx, self.r2_idx]
        self.r_name_list = [self.r_name1, self.r_name2]

        self.eval_folder = "./TwoCons_eval/" + "{}_{}/".format(r_name1.replace("dbo:", ""), r_name2.replace("dbo:", ""))

    def load_rules_and_model(self, r_name):
        self.logger.info("Collect rules, R: {}".format(r_name))
        if r_name not in self.r_model_dict:
            graph.rule_file = "../../MyData/DBO/United_States/model/{}/rule.txt".format(r_name.split(":")[-1])
            graph.model_file = "../../MyData/DBO/United_States/model/{}/{}_model.tar".format(r_name.split(":")[-1],
                                                                                             pca_or_cwa)
            graph.rule_num_to_use_file = "../../MyData/DBO/United_States/model/{}/rule_num_to_use.txt".format(
                r_name.split(":")[-1])

            self.r_rules_dict[r_name] = self.graph.load_rule_obj_from_file(r_name)
            self.r_model_dict[r_name] = self.graph.get_pra_model4r(self.graph.r_id_by_name(r_name))

    def split_cons(self, cons, direction):
        if direction == "left":
            first_r_name, first_e_name = cons.split()
        else:
            first_e_name, first_r_name = cons.split()
        return first_e_name, first_r_name

    def load_constraints(self, direction1, direction2):
        test_examples_file = "{}/2consQandA.txt".format(self.eval_folder)
        first_e_idx_list = []
        first_r_idx_list = []
        second_e_idx_list = []
        second_r_idx_list = []
        answer_idx_list = []
        with open(test_examples_file, 'r', encoding="UTF-8") as f:
            all_lines = f.readlines()
            cnt = 0
            while True:
                if cnt >= len(all_lines) or all_lines[cnt].strip() == "":
                    break
                first_e_name, first_r_name = self.split_cons(all_lines[cnt], direction1)
                first_e_idx_list.append(self.graph.get_e_idx_by_e_name(first_e_name))
                first_r_idx_list.append(self.graph.r_id_by_name(first_r_name))
                cnt += 1
                second_e_name, second_r_name = self.split_cons(all_lines[cnt], direction2)
                second_e_idx_list.append(self.graph.get_e_idx_by_e_name(second_e_name))
                second_r_idx_list.append(self.graph.r_id_by_name(second_r_name))
                cnt += 1
                answer_name = all_lines[cnt].split()
                answer_idx_list.append([self.graph.get_e_idx_by_e_name(name) for name in answer_name])
                cnt += 1
        return first_e_idx_list, first_r_idx_list, second_e_idx_list, second_r_idx_list, answer_idx_list

    def get_ap_rr_hits10_firstRank(self, res_prob_list, answer_list):
        cnt = 0
        rr = 0.0
        ap = 0.0
        hits10 = 0
        first_rank = 0
        for res_ht_idx, cand_prob in enumerate(res_prob_list):
            cand_idx = cand_prob[0]
            if cand_idx in answer_list:
                if rr == 0:
                    rr = 1.0 / (1 + res_ht_idx)
                    first_rank = (1 + res_ht_idx)
                if res_ht_idx < 10:
                    hits10 = 1
                cnt += 1
                ap += cnt / (res_ht_idx + 1)

        if cnt != 0:
            ap /= cnt
        return ap, rr, hits10, first_rank

    def pra_map_mrr_hits10_avg_time(self):
        for idx, r_name in enumerate(self.r_name_list):
            self.load_rules_and_model(r_name)

        first_e_idx_list, first_r_idx_list, second_e_idx_list, second_r_idx_list, answer_idx_list \
            = self.load_constraints(self.r1_direction, self.r2_direction)

        hits10_r = 0.0
        map_r = 0.0
        mrr_r = 0.0
        mean_rank_r = 0
        metric_file = "{}{}_rdqe_metric.txt".format(self.eval_folder, pca_or_cwa)
        average_time = 0.0

        def convert2dict(tmp_res_list):
            tmp_dict = {}
            for tmp_res in tmp_res_list:
                prob_val = tmp_res[1]
                tar_idx = tmp_res[0]
                tmp_dict[tar_idx] = prob_val
            return tmp_dict

        def get_1cons_res(e_idx, r_idx, direction):
            r_name = self.graph.get_r_name_by_r_idx(r_idx)
            r_rules = self.r_rules_dict[r_name]
            r_model = self.r_model_dict[r_name]
            if direction == "left":
                tmp_res_prob_list = self.graph.get_head_from_tail_by_rule(e_idx, r_rules, r_model)
            else:
                tmp_res_prob_list = self.graph.get_tail_from_head_by_rule(e_idx, r_rules, r_model)
            return tmp_res_prob_list

        for i in range(len(first_r_idx_list)):
            e1_idx = first_e_idx_list[i]
            r1_idx = first_r_idx_list[i]
            e2_idx = second_e_idx_list[i]
            r2_idx = second_r_idx_list[i]

            start_time = time.time()

            res1_prob_list = get_1cons_res(e1_idx, r1_idx, self.r1_direction)
            res1_prob_dict = convert2dict(res1_prob_list)

            res2_prob_list = get_1cons_res(e2_idx, r2_idx, self.r2_direction)
            res2_prob_dict = convert2dict(res2_prob_list)

            res_prob_list = []
            for tar_idx in res1_prob_dict:
                if tar_idx in res2_prob_dict:
                    val = res2_prob_dict[tar_idx] * res1_prob_dict[tar_idx]
                    res_prob_list.append([tar_idx, val])

            res_prob_list.sort(key=lambda x: x[1], reverse=True)
            end_time = time.time()
            print("{}/{}, Time elapsed: {}".format(i + 1, len(first_r_idx_list), end_time - start_time))
            average_time += end_time - start_time

            ap, rr, hits10, first_rank = self.get_ap_rr_hits10_firstRank(res_prob_list, answer_idx_list[i])

            map_r += ap
            mrr_r += rr
            hits10_r += hits10
            mean_rank_r += first_rank

        mrr_r = mrr_r / len(first_r_idx_list)
        hits10_r /= len(first_r_idx_list)
        map_r /= len(first_r_idx_list)
        mean_rank_r /= len(first_r_idx_list)
        average_time /= len(first_r_idx_list)

        print("mrr:{}\thits10:{}\tmap:{}\tmean rank:{}\taverage time: {}".format(mrr_r, hits10_r, map_r, mean_rank_r,
                                                                                 average_time))

        with open(metric_file, 'w', encoding="UTF-8") as f:
            f.write("hits10\t{}\n".format(hits10_r))
            f.write("map\t{}\n".format(map_r))
            f.write("mrr\t{}\n".format(mrr_r))
            f.write("average time\t{}\n".format(average_time))


if __name__ == "__main__":

    search_folder = "../../MyData/DBO/All/"
    graph = Graph()

    graph.e2idx_file = search_folder + "entity2id.txt"
    graph.r2idx_file = search_folder + "relation2id.txt"
    graph.triple2idx_file = search_folder + "triple2id.txt"

    graph.load_data()

    two_conts_eval = TwoConts()
    two_conts_eval._setGraph(graph)

    # r_name_list_right_left = [["dbo:starring", "dbo:birthPlace", ]]
    #
    # for r_name in r_name_list_right_left:
    #     print("Right Left")
    #     print("R_name:{}\tR_name:{}".format(r_name[0], r_name[1]))
    #     two_conts_eval._set_r_name(r_name[0], "right", r_name[1], "left")
    #     two_conts_eval.pra_map_mrr_hits10_avg_time()

    # r_name_list_right = [["dbo:residence", "dbo:deathPlace"]]
    # for r_name in r_name_list_right:
    #     print("Right")
    #     print("R_name:{}\tR_name:{}".format(r_name[0], r_name[1]))
    #     two_conts_eval._set_r_name(r_name[0], "right", r_name[1], "right")
    #     two_conts_eval.pra_map_mrr_hits10_avg_time()

    # r_name_list_left = [["dbo:product", "dbo:foundationPlace"],
    #                     ["dbo:regionServed", "dbo:location"],
    #                     ["dbo:birthPlace", "dbo:award"]]
    # r_name_list_left = [["dbo:regionServed", "dbo:location"]]

    # r_name_list_left = [["dbo:owner", "dbo:foundationPlace"],
    #                     ["dbo:regionServed", "dbo:product"],
    #                     ["dbo:locationCountry", "dbo:foundationPlace"],
    #                     ["dbo:regionServed", "dbo:owner"]]

    r_name_list_left = [["dbo:locationCountry", "dbo:foundationPlace"],
                        ["dbo:regionServed", "dbo:owner"]]

    for r_name in r_name_list_left:
        print("Left")
        print("R_name:{}\tR_name:{}".format(r_name[0], r_name[1]))
        two_conts_eval._set_r_name(r_name[0], "left", r_name[1], "left")
        two_conts_eval.pra_map_mrr_hits10_avg_time()

