from RuleBased.Params import file_path_seg
import json
import numpy as np


class TransEMAP:
    def __init__(self, relation):
        self.example_folder = "F:\\Data\\dbpedia\\"
        self.embedding_scope = "United_States_Canada"
        self.embedding_file = self.example_folder + self.embedding_scope + "\\TransE.json"
        self.entity2id_file = self.example_folder + self.embedding_scope + "\\entity2id.txt"
        self.relation2id_file = self.example_folder + self.embedding_scope + "\\relation2id.txt"

        self.test_scope = "Canada" + file_path_seg
        self.test_folder = self.example_folder + self.test_scope + "test_model" + file_path_seg

        self.relation = relation
        self.test_folder = self.test_folder + self.relation.split(":")[-1] + file_path_seg

        self.posi_ht_file = self.test_folder + "posi_ht.txt"
        self.nege_ht_file = self.test_folder + "nege_ht.txt"

        self.posi_ht_list = []
        self.nege_ht_list = []

        self.idx2e = {}
        self.idx2r = {}
        self.e2idx = {}
        self.r2idx = {}

        self.canada_folder = "F:\\Data\\dbpedia\\Canada\\"
        self.old_idx2e = {}
        self.old_e2idx = {}
        self.old_idx2r = {}
        self.old_r2idx = {}

        self.e_embed = []
        self.r_embed = []

    def load_data(self):
        self.load_er()
        self.load_ht()
        self.load_embedding()

    def load_er(self):
        with open(self.entity2id_file,'r',encoding="UTF-8") as f:
            for line in f.readlines():
                line_array = line.split()
                if len(line_array) == 1:
                    continue
                self.idx2e[int(line_array[1])] = line_array[0]
                self.e2idx[line_array[0]] = int(line_array[1])

        with open(self.relation2id_file,'r',encoding="UTF-8") as f:
            for line in f.readlines():
                line_array = line.split()
                if len(line_array) == 1:
                    continue
                self.idx2r[int(line_array[1])] = line_array[0]
                self.r2idx[line_array[0]] = int(line_array[1])

        with open(self.canada_folder+"e2idx_shortcut.txt",'r',encoding="UTF-8") as f:
            for line in f.readlines():
                num,name = line.split()
                self.old_e2idx[name] = int(num)
                self.old_idx2e[int(num)] = name

        with open(self.canada_folder+"r2idx_shortcut.txt","r",encoding="UTF-8") as f:
            for line in f.readlines():
                num,name = line.split()
                self.old_r2idx[name] = int(num)
                self.old_idx2r[int(num)] = name

    def load_ht(self):
        with open(self.posi_ht_file, 'r', encoding="UTF-8") as f:
            for idx, line in enumerate(f.readlines()):
                if idx == 0:
                    continue
                ht_idx = [int(num) for num in line.split()]
                h_name = self.old_idx2e[ht_idx[0]]
                t_name = self.old_idx2e[ht_idx[1]]
                new_h_idx = self.e2idx[h_name]
                new_t_idx = self.e2idx[t_name]
                self.posi_ht_list.append([new_h_idx,new_t_idx ])

        with open(self.nege_ht_file, 'r', encoding="UTF-8") as f:
            for idx, line in enumerate(f.readlines()):
                if idx == 0:
                    continue
                ht_idx = [int(num) for num in line.split()]
                h_name = self.old_idx2r[ht_idx[0]]
                t_name = self.old_idx2r[ht_idx[1]]
                new_h_idx = self.r2idx[h_name]
                new_t_idx = self.r2idx[t_name]
                self.nege_ht_list.append([new_h_idx, new_t_idx])

    def load_embedding(self):
        with open(self.embedding_file, 'r', encoding="UTF-8") as f:
            content = json.loads(f.read())
            self.e_embed = np.array(content['ent_embeddings.weight'])
            self.r_embed = np.array(content['rel_embeddings.weight'])

    def get_map(self):
        new_rel_idx = self.r2idx[self.relation]

        label_list = [1] * len(self.posi_ht_list)
        label_list.extend([0] * len(self.nege_ht_list))

        self.posi_ht_list.extend(self.nege_ht_list)
        r_list = [new_rel_idx] * len(self.posi_ht_list)

        h_list = [ht[0] for ht in self.posi_ht_list]
        t_list = [ht[1] for ht in self.posi_ht_list]

        output = np.linalg.norm(self.e_embed[h_list] + self.r_embed[r_list] - self.e_embed[t_list])
        res = list(zip(output,label_list))
        res.sort(key=lambda x: x[0])
        map_metric = 0.0
        cnt = 0
        for idx,a_pair in enumerate(res):
            if a_pair[1] == 1:
                cnt += 1
                map_metric += 1.0 / cnt
        with open("F:\\Data\\dbpedia\\Canada\\test_model\\" +self.relation.split()[-1] + "\\transe_metric",'w',) as f:
            f.write("MAP: {}\n".format(map_metric/cnt))






if __name__ == "__main__":
    relation = "dbo:locationCountry"
    tm = TransEMAP(relation)
    tm.load_embedding()

