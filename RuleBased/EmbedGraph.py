import numpy as np
import torch.nn as nn
import torch
import os
import numpy as np
import time
import json

from RuleBased.Params import transe_embed

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'


class EGraph(nn.Module):
    def __init__(self):
        super(EGraph, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if transe_embed:
            self._load_data()
        else:
            self._load_data2()

    def _load_data(self):

        self.usa_transe = "../MyData/DBO/United_States/TransE.json"
        with open(self.usa_transe, 'r', encoding="UTF-8") as f:
            te = json.load(f)
        self.e2vec = te['ent_embeddings.weight']
        self.r2vec = te['rel_embeddings.weight']

        self.ent_embeds = nn.Embedding.from_pretrained(torch.FloatTensor(self.e2vec))
        self.rel_embeds = nn.Embedding.from_pretrained(torch.FloatTensor(self.r2vec))
        self.cos_calculator = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.all_ent_idx = torch.LongTensor([i for i in range(len(self.e2vec))]).to(self.device)
        self.my_all_embedding = self.ent_embeds(self.all_ent_idx)
        # self.my_all_embedding = torch.transpose(self.ent_embeddings(self.all_ent_idx), 0, 1)
        self.to(self.device)

    def _load_data2(self):

        self.e2vec = []
        self.r2vec = []
        self.folder = "../../MyData/EmbedDBO/"
        self.e2vec_file = self.folder + "entity2vector.txt"
        self.r2vec_file = self.folder + "relation2vector.txt"
        with open(self.e2vec_file, 'r', encoding="UTF-8") as f:
            for line in f.readlines():
                self.e2vec.append([float(num) for num in line.split()])
            self.e2vec = np.array(self.e2vec)

        with open(self.r2vec_file, 'r', encoding="UTF-8") as f:
            for line in f.readlines():
                self.r2vec.append([float(num) for num in line.split()])
            self.r2vec = np.array(self.r2vec)

        self.ent_embeds = nn.Embedding.from_pretrained(torch.FloatTensor(self.e2vec))
        self.rel_embeds = nn.Embedding.from_pretrained(torch.FloatTensor(self.r2vec))
        self.cos_calculator = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.all_ent_idx = torch.LongTensor([i for i in range(len(self.e2vec))]).to(self.device)
        self.my_all_embedding = self.ent_embeds(self.all_ent_idx)
        # self.my_all_embedding = torch.transpose(self.ent_embeddings(self.all_ent_idx), 0, 1)
        self.to(self.device)

    def get_e_embed(self, e_idx):
        return self.e2vec[e_idx]

    def get_r_embed(self, r_idx):
        return self.r2vec[r_idx]

    def get_triple_conf_sum(self, ht_idx_list, r_idx):
        cos_simi = self.get_every_triple_conf(ht_idx_list, r_idx)
        return np.sum(cos_simi)

    def get_every_triple_conf(self, ht_idx_list, r_idx):
        h_idx_list_tensor = torch.LongTensor([int(ht[0]) for ht in ht_idx_list]).to(self.device)
        t_idx_list_tensor = torch.LongTensor([int(ht[1]) for ht in ht_idx_list]).to(self.device)
        r_idx_list_tensor = torch.LongTensor([int(r_idx)] * len(ht_idx_list)).to(self.device)
        tmp = self.ent_embeds(h_idx_list_tensor) + self.rel_embeds(r_idx_list_tensor)
        cos_simi = self.cos_calculator(tmp, self.ent_embeds(t_idx_list_tensor)).cpu().numpy()
        cos_simi = (cos_simi + 1.0) / 2.0
        return cos_simi

    def map_hits10_rr_firstRank(self, t_ranking_list):
        t_ranking_list.sort()

        ap = 0.0
        rr = 0
        hits10 = 0

        for idx, t_pos in enumerate(t_ranking_list):
            ap += (idx + 1) / (1.0 * (t_pos + 1))
            if idx == 0:
                rr = 1.0 / (1.0 * (t_pos + 1))
            if t_pos < 10:
                hits10 = 1

        return ap / len(t_ranking_list), hits10, rr, t_ranking_list[0]

    def get_map_mrr_hits10_meanRank_avg_time_embed_1cons(self, h_idx_list, r_idx_list, t_idx_list):
        start_time = time.time()

        map_r = 0.0
        hits10_r = 0
        mrr_r = 0.0
        mean_rank = 0
        time_elapsed = 0

        h_idx_list_tensor = torch.LongTensor(h_idx_list).to(self.device)
        r_idx_list_tensor = torch.LongTensor(r_idx_list).to(self.device)

        add_result_tensor = self.ent_embeds(h_idx_list_tensor) + self.rel_embeds(r_idx_list_tensor)
        for idx, res_embedding in enumerate(add_result_tensor):
            cos_simi = self.cos_calculator(res_embedding.unsqueeze(0), self.my_all_embedding).cpu().numpy()
            t_start_time = time.time()
            arg_ranking = list(np.argsort(-cos_simi))
            t_ranking_list = []
            for t_idx in t_idx_list[idx]:
                t_ranking_list.append(arg_ranking.index(t_idx))

            ap, hits10, rr, first_rank = self.map_hits10_rr_firstRank(t_ranking_list)

            hits10_r += hits10
            map_r += ap
            mrr_r += rr
            mean_rank += first_rank
            print("{}/{}".format(idx, len(h_idx_list)))
            print("ap:{}\trr:{}\thits10:{}\tfirst rank:{}".format(ap, rr, hits10, first_rank))
            t_end_time = time.time()
            time_elapsed -= t_end_time - t_start_time
        end_time = time.time()
        time_elapsed = (end_time - start_time + time_elapsed) / len(h_idx_list)
        return map_r / len(h_idx_list), mrr_r / len(h_idx_list), hits10_r / len(h_idx_list), mean_rank / len(
            h_idx_list), time_elapsed

    # def get_map_mrr_hits10_meanRank_embed_2cons_right_var(self, f_h_idx_list, f_r_idx_list,
    #                                                       s_h_idx_list, s_r_idx_list,
    #                                                       t_idx_list):
    #     map_r = 0.0
    #     hits10_r = 0
    #     mrr_r = 0.0
    #     mean_rank = 0
    #     data_len = len(f_h_idx_list)
    #
    #     f_h_idx_list_tensor = torch.LongTensor(f_h_idx_list).to(self.device)
    #     f_r_idx_list_tensor = torch.LongTensor(f_r_idx_list).to(self.device)
    #
    #     s_h_idx_list_tensor = torch.LongTensor(s_h_idx_list).to(self.device)
    #     s_r_idx_list_tensor = torch.LongTensor(s_r_idx_list).to(self.device)
    #
    #     add_result_tensor = 0.5 * (self.ent_embeds(f_h_idx_list_tensor)
    #                                + self.rel_embeds(f_r_idx_list_tensor)
    #                                + self.ent_embeds(s_h_idx_list_tensor)
    #                                + self.rel_embeds(s_r_idx_list_tensor))
    #
    #     for idx, res_embedding in enumerate(add_result_tensor):
    #         cos_simi = self.cos_calculator(res_embedding.unsqueeze(0), self.my_all_embedding).cpu().numpy()
    #         arg_ranking = list(np.argsort(-cos_simi))
    #         t_ranking_list = []
    #         for t_idx in t_idx_list[idx]:
    #             t_ranking_list.append(arg_ranking.index(t_idx))
    #         ap, hits10, rr, first_rank = self.map_hits10_rr_firstRank(t_ranking_list)
    #
    #         hits10_r += hits10
    #         map_r += ap
    #         mrr_r += rr
    #         mean_rank += first_rank
    #         print("{}/{}".format(idx, data_len))
    #         print("ap:{}\trr:{}\thits10:{}\tfirst rank:{}".format(ap, rr, hits10, first_rank))
    #     return map_r / data_len, mrr_r / data_len, hits10_r / data_len, mean_rank / data_len
    #
    # def get_map_mrr_hits10_meanRank_embed_2cons_left_var(self, f_h_idx_list, f_r_idx_list,
    #                                                      s_h_idx_list, s_r_idx_list,
    #                                                      t_idx_list):
    #     map_r = 0.0
    #     hits10_r = 0
    #     mrr_r = 0.0
    #     mean_rank = 0
    #     data_len = len(f_h_idx_list)
    #
    #     f_h_idx_list_tensor = torch.LongTensor(f_h_idx_list).to(self.device)
    #     f_r_idx_list_tensor = torch.LongTensor(f_r_idx_list).to(self.device)
    #
    #     s_h_idx_list_tensor = torch.LongTensor(s_h_idx_list).to(self.device)
    #     s_r_idx_list_tensor = torch.LongTensor(s_r_idx_list).to(self.device)
    #
    #     add_result_tensor = 0.5 * (self.ent_embeds(f_h_idx_list_tensor)
    #                                - self.rel_embeds(f_r_idx_list_tensor)
    #                                + self.ent_embeds(s_h_idx_list_tensor)
    #                                - self.rel_embeds(s_r_idx_list_tensor))
    #
    #     for idx, res_embedding in enumerate(add_result_tensor):
    #         cos_simi = self.cos_calculator(res_embedding.unsqueeze(0), self.my_all_embedding).cpu().numpy()
    #         arg_ranking = list(np.argsort(-cos_simi))
    #         t_ranking_list = []
    #         for t_idx in t_idx_list[idx]:
    #             t_ranking_list.append(arg_ranking.index(t_idx))
    #         ap, hits10, rr, first_rank = self.map_hits10_rr_firstRank(t_ranking_list)
    #
    #         hits10_r += hits10
    #         map_r += ap
    #         mrr_r += rr
    #         mean_rank += first_rank
    #         print("{}/{}".format(idx, data_len))
    #         print("ap:{}\trr:{}\thits10:{}\tfirst rank:{}".format(ap, rr, hits10, first_rank))
    #     return map_r / data_len, mrr_r / data_len, hits10_r / data_len, mean_rank / data_len

    def get_projection_embedding(self, e_embedding, r_embedding, direction):
        if direction == "left":
            return e_embedding - r_embedding
        else:
            return e_embedding + r_embedding

    def get_map_mrr_hits10_meanRank_embed_2cons(self,
                                                f_h_idx_list, f_r_idx_list, direction1,
                                                s_h_idx_list, s_r_idx_list, direction2,
                                                t_idx_list):
        map_r = 0.0
        hits10_r = 0
        mrr_r = 0.0
        mean_rank = 0
        data_len = len(f_h_idx_list)

        f_h_idx_list_tensor = torch.LongTensor(f_h_idx_list).to(self.device)
        f_r_idx_list_tensor = torch.LongTensor(f_r_idx_list).to(self.device)

        s_h_idx_list_tensor = torch.LongTensor(s_h_idx_list).to(self.device)
        s_r_idx_list_tensor = torch.LongTensor(s_r_idx_list).to(self.device)

        cons1_tensor = self.get_projection_embedding(self.ent_embeds(f_h_idx_list_tensor)
                                                     , self.rel_embeds(f_r_idx_list_tensor), direction1)
        cons2_tensor = self.get_projection_embedding(self.ent_embeds(s_h_idx_list_tensor)
                                                     , self.rel_embeds(s_r_idx_list_tensor), direction2)
        add_result_tensor = 0.5 * (cons1_tensor + cons2_tensor)

        for idx, res_embedding in enumerate(add_result_tensor):
            cos_simi = self.cos_calculator(res_embedding.unsqueeze(0), self.my_all_embedding).cpu().numpy()
            arg_ranking = list(np.argsort(-cos_simi))
            t_ranking_list = []
            for t_idx in t_idx_list[idx]:
                t_ranking_list.append(arg_ranking.index(t_idx))
            ap, hits10, rr, first_rank = self.map_hits10_rr_firstRank(t_ranking_list)

            hits10_r += hits10
            map_r += ap
            mrr_r += rr
            mean_rank += first_rank
            print("{}/{}".format(idx, data_len))
            print("ap:{}\trr:{}\thits10:{}\tfirst rank:{}".format(ap, rr, hits10, first_rank))
        return map_r / data_len, mrr_r / data_len, hits10_r / data_len, mean_rank / data_len

    # def get_map_mrr_hits10_meanRank_embed_2cons_right_left_var(self, f_h_idx_list, f_r_idx_list,
    #                                                            s_h_idx_list, s_r_idx_list,
    #                                                            t_idx_list):
    #     map_r = 0.0
    #     hits10_r = 0
    #     mrr_r = 0.0
    #     mean_rank = 0
    #     data_len = len(f_h_idx_list)
    #
    #     f_h_idx_list_tensor = torch.LongTensor(f_h_idx_list).to(self.device)
    #     f_r_idx_list_tensor = torch.LongTensor(f_r_idx_list).to(self.device)
    #
    #     s_h_idx_list_tensor = torch.LongTensor(s_h_idx_list).to(self.device)
    #     s_r_idx_list_tensor = torch.LongTensor(s_r_idx_list).to(self.device)
    #
    #     add_result_tensor = 0.5 * (self.ent_embeds(f_h_idx_list_tensor)
    #                                + self.rel_embeds(f_r_idx_list_tensor)
    #                                + self.ent_embeds(s_h_idx_list_tensor)
    #                                - self.rel_embeds(s_r_idx_list_tensor))
    #
    #     for idx, res_embedding in enumerate(add_result_tensor):
    #         cos_simi = self.cos_calculator(res_embedding.unsqueeze(0), self.my_all_embedding).cpu().numpy()
    #         arg_ranking = list(np.argsort(-cos_simi))
    #         t_ranking_list = []
    #         for t_idx in t_idx_list[idx]:
    #             t_ranking_list.append(arg_ranking.index(t_idx))
    #         ap, hits10, rr, first_rank = self.map_hits10_rr_firstRank(t_ranking_list)
    #
    #         hits10_r += hits10
    #         map_r += ap
    #         mrr_r += rr
    #         mean_rank += first_rank
    #         print("{}/{}".format(idx, data_len))
    #         print("ap:{}\trr:{}\thits10:{}\tfirst rank:{}".format(ap, rr, hits10, first_rank))
    #     return map_r / data_len, mrr_r / data_len, hits10_r / data_len, mean_rank / data_len

#
# if __name__ == "__main__":
#     ht_id_list = [[1, 2], [3, 4], [5, 6]]
#     r_idx = 4
#     eg = EGraph()
#     cos_simi = eg.get_every_triple_conf(ht_id_list, r_idx)
#     print(cos_simi)
