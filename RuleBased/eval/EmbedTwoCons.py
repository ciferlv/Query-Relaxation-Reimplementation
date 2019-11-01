from RuleBased.Graph import Graph
from RuleBased.FilePathConfig import FilePathConfig
import time


class GetMetricEmbedding:
    def __init__(self):
        self.graph = Graph()
        self.graph.e2idx_file = "../../MyData/DBO/All/entity2id.txt"
        self.graph.r2idx_file = "../../MyData/DBO/All/relation2id.txt"
        self.graph.triple2idx_file = "../../MyData/DBO/All/triple2id.txt"
        self.graph.load_data()
        self.graph.load_er_embedding()

    def split_cons(self, cons, direction):
        if direction == "left":
            first_r_name, first_e_name = cons.split()
        else:
            first_e_name, first_r_name = cons.split()
        return first_e_name, first_r_name

    def get_metric(self, r_name1, direction1, r_name2, direction2):
        example_file_path = "./TwoCons_eval/{}_{}/2consQandA.txt".format(r_name1.replace("dbo:", ""),
                                                                           r_name2.replace("dbo:", ""))
        metric_file = "./TwoCons_eval/{}_{}/metric.txt".format(r_name1.replace("dbo:", ""),
                                                                 r_name2.replace("dbo:", ""))
        first_e_idx_list = []
        first_r_idx_list = []
        second_e_idx_list = []
        second_r_idx_list = []
        answer_idx_list = []
        with open(example_file_path, 'r', encoding="UTF-8") as f:
            all_lines = f.readlines()
            cnt = 0
            while True:
                if cnt >= len(all_lines) or all_lines[cnt].strip() == "":
                    break
                first_e_name, first_r_name = self.split_cons(all_lines[cnt], direction1)
                first_e_idx_list.append(self.graph.e_name2id[first_e_name])
                first_r_idx_list.append(self.graph.r_name2id[first_r_name])
                cnt += 1
                second_e_name, second_r_name = self.split_cons(all_lines[cnt], direction2)
                second_e_idx_list.append(self.graph.e_name2id[second_e_name])
                second_r_idx_list.append(self.graph.r_name2id[second_r_name])
                cnt += 1
                answer_name = all_lines[cnt].split()
                answer_idx_list.append([self.graph.e_name2id[name] for name in answer_name])
                cnt += 1
        start_time = time.time()
        map_r, mrr_r, hits10_r, mean_rank = self.graph.ekg.get_map_mrr_hits10_meanRank_embed_2cons(
            first_e_idx_list, first_r_idx_list, direction1,
            second_e_idx_list, second_r_idx_list, direction2,
            answer_idx_list)
        end_time = time.time()
        with open(metric_file, 'w', encoding="UTF-8") as f:
            f.write("map:{}\n".format(map_r))
            f.write("mrr:{}\n".format(mrr_r))
            f.write("hits10:{}\n".format(hits10_r))
            f.write("mean_rank:{}\n".format(mean_rank))
            f.write("average_time:{}\n".format((end_time - start_time) / len(first_e_idx_list)))


if __name__ == "__main__":
    # r_name_list_left = [["dbo:product", "dbo:foundationPlace"],
    #                     ["dbo:regionServed", "dbo:location"],
    #                     ["dbo:birthPlace", "dbo:award"]]
    # r_name_list_right = [["dbo:residence", "dbo:deathPlace"]]
    # r_name_list_right_left = [["dbo:starring", "dbo:birthPlace"]]

    r_name_list_left = [["dbo:owner", "dbo:foundationPlace"],
                        ["dbo:regionServed", "dbo:product"],
                        ["dbo:locationCountry", "dbo:foundationPlace"],
                        ["dbo:regionServed", "dbo:owner"]]

    gm = GetMetricEmbedding()

    # for r_name in r_name_list_right_left:
    #     print("Right Left")
    #     print("R_name:{}\tR_name:{}".format(r_name[0], r_name[1]))
    #     gm.get_metric(r_name[0], "right", r_name[1], "left")

    # for r_name in r_name_list_right:
    #     print("Right")
    #     print("R_name:{}\tR_name:{}".format(r_name[0], r_name[1]))
    #     gm.get_metric(r_name[0], "right", r_name[1], "right")

    for r_name in r_name_list_left:
        print("Left")
        print("R_name:{}\tR_name:{}".format(r_name[0], r_name[1]))
        gm.get_metric(r_name[0], "left", r_name[1], "left")
