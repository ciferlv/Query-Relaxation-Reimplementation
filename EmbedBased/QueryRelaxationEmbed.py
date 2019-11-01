import numpy as np
from Empty_Answer_Query import eaqs
from RuleBased.SparqlParser import SparqlParser


class QRE:
    def __init__(self):
        self.folder = "../MyData/EmbedDBO/"
        self.e2idx_file = self.folder + "entity2id.txt"
        self.r2idx_file = self.folder + "relation2id.txt"
        self.e2vec_file = self.folder + "entity2vector.txt"
        self.r2vec_file = self.folder + "relation2vector.txt"

        self.e2vec = []
        self.r2vec = []

        self.e2idx = {}
        self.idx2e = {}
        self.r2idx = {}
        self.idx2r = {}

    def load_data(self):

        with open(self.e2vec_file, 'r', encoding="UTF-8") as f:
            for line in f.readlines():
                self.e2vec.append([float(num) for num in line.split()])
            self.e2vec = np.array(self.e2vec)

        with open(self.r2vec_file, 'r', encoding="UTF-8") as f:
            for line in f.readlines():
                self.r2vec.append([float(num) for num in line.split()])
            self.r2vec = np.array(self.r2vec)

        with open(self.e2idx_file, 'r', encoding="UTF-8") as f:
            for line in f.readlines()[1:]:
                name,idx  = line.strip().split()
                # name = getShortcutName(name)
                idx = int(idx)
                self.e2idx[name] = idx
                self.idx2e[idx] = name

        with open(self.r2idx_file, 'r', encoding="UTF-8") as f:
            for line in f.readlines()[1:]:
                name,idx = line.strip().split()
                # name = getShortcutName(name)
                idx = int(idx)
                self.r2idx[name] = idx
                self.idx2r[idx] = name

    def get_embed1(self, r_idx, e_idx, o1):
        res = [0] * 50
        if o1 == 'left':
            res = self.e2vec[e_idx] + self.r2vec[r_idx]
        elif o1 == "right":
            res = self.e2vec[e_idx] - self.r2vec[r_idx]
        return res

    def get_similar_e(self, res_vec):
        similar_entity = []
        topK = 10
        simi_list = -1 * (np.dot(self.e2vec, res_vec) / (np.linalg.norm(res_vec) * np.linalg.norm(self.e2vec, axis=-1)))
        sorted_index = np.argsort(simi_list)
        res_str = ""
        for i in range(topK):
            res_str += "{}/{}.\n".format(i + 1, topK)
            min_idx = sorted_index[i]
            res_str += "e_idx: {}, e_name: {}, simi: {}\n".format(min_idx, self.idx2e[min_idx], -simi_list[min_idx])
            similar_entity.append(self.idx2e[min_idx])
        return res_str, similar_entity

    def check_all_query(self):
        # execute_list = [0]
        for idx, sparql_query in enumerate(eaqs):
            if idx != 16:
                continue
            print("Get Reulst of : {}".format(sparql_query.strip()))
            var_vec_dict = {}
            sp = SparqlParser(sparql_query)
            sp.parse_sparql()
            for var in sp.var1BGP:
                res_vec = [0] * 50
                for h, r, t in sp.var1BGP[var]:
                    r_idx = self.r2idx[r]
                    if "?" in h:
                        o = "right"
                        e_idx = self.e2idx[t]
                    else:
                        o = "left"
                        e_idx = self.e2idx[h]
                    res_vec += self.get_embed1(r_idx, e_idx, o)
                res_vec /= len(sp.var1BGP[var])
                var_vec_dict[var] = res_vec
            for var in sp.var2BGP:
                res_vec = [0] * 50
                tar_var = ""
                for h, r, t in sp.var2BGP[var]:
                    r_idx = self.r2idx[r]
                    if h not in var_vec_dict:
                        res_vec += var_vec_dict[t] - self.r2vec[r_idx]
                        tar_var = h
                    else:
                        res_vec += var_vec_dict[h] + self.r2vec[r_idx]
                        tar_var = t
                res_vec /= len(sp.var2BGP)
                var_vec_dict[tar_var] = res_vec

            res_str = ""
            for var in var_vec_dict:
                res_str += var + "\n"
                var_vec = var_vec_dict[var]
                tmp_str, similar_entity_list = self.get_similar_e(var_vec)
                res_str += tmp_str + "\n"
            with open("./result/E" + str(idx + 1) + ".txt", 'w', encoding="UTF-8") as f:
                f.write(sparql_query + "\n")
                f.write(res_str + "\n")


if __name__ == "__main__":
    qre = QRE()
    qre.load_data()
    qre.check_all_query()
