import numpy as np

folder = "F:\\Data\\dbpedia\\EmbedData\\"
e2idx_file = folder + "e2id.txt"
r2idx_file = folder + "r2id.txt"
e2vec_file = folder + "entity2vector.txt"
r2vec_file = folder + "relation2vector.txt"

e2vec = []
with open(e2vec_file, 'r', encoding="UTF-8") as f:
    for line in f.readlines():
        e2vec.append([float(num) for num in line.split()])
    e2vec = np.array(e2vec)

r2vec = []
with open(r2vec_file, 'r', encoding="UTF-8") as f:
    for line in f.readlines():
        r2vec.append([float(num) for num in line.split()])
    r2vec = np.array(r2vec)

e2idx = {}
idx2e = {}
r2idx = {}
idx2r = {}

with open(e2idx_file, 'r', encoding="UTF-8") as f:
    for line in f.readlines():
        idx, name = line.split()
        idx = int(idx)
        e2idx[name] = idx
        idx2e[idx] = name

with open(r2idx_file, 'r', encoding="UTF-8") as f:
    for line in f.readlines():
        idx, name = line.split()
        idx = int(idx)
        r2idx[name] = idx
        idx2r[idx] = name

if __name__ == "__main__":
    e1_name = "<http://dbpedia.org/resource/Isaac_Newton>"
    r_name = "<http://dbpedia.org/ontology/doctoralAdvisor>"
    e2_name = ""
    e1_idx = e2idx[e1_name]
    r_idx = r2idx[r_name]
    e1_vec = e2vec[e1_idx]
    r_vec = r2vec[r_idx]
    res_vec = np.array(e1_vec + r_vec)
    simi = np.dot(e2vec, res_vec) / (np.linalg.norm(res_vec) * np.linalg.norm(e2vec, axis=-1))
    sorted_index = np.argsort(simi)
    for i in range(10):
        print("{}/{}.".format(i+1, 10))
        min_idx = sorted_index[i]
        print(min_idx)
        print(idx2e[min_idx])
