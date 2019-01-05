# rdf_file = "F:\Data\dbpedia\mappingbased_objects_en.ttl"
rdf_file = "./data/FTEST"
e2idx_file = "./data/e2idx.txt"
r2idx_file = "./data/r2idx.txt"
triple2idx_file = "./data/triple2idx.txt"
statistics_file = "./data/statistics.txt"


def data2idx():
    e2idx = {}
    r2idx = {}
    triple_list = []
    e_cnt = 0
    r_cnt = 0
    triple_cnt = 0
    with open(rdf_file, "r", encoding="UTF-8") as f:
        for idx, line in enumerate(f.readlines()):
            if line.strip().startswith("#"):
                print(line.strip())
                continue
            h, r, t = line.strip().strip(".").split()
            if h not in e2idx:
                e2idx[h] = e_cnt
                e_cnt += 1
            if r not in r2idx:
                r2idx[r] = r_cnt
                r_cnt += 1
            if t not in e2idx:
                e2idx[t] = e_cnt
                e_cnt += 1
            triple_list.append([e2idx[h], r2idx[r], e2idx[t]])
            triple_cnt += 1

    with open(statistics_file, 'w', encoding="UTF-8") as f:
        f.write("Entity Num: {}\n".format(e_cnt))
        f.write("Relation Num: {}\n".format(r_cnt))
        f.write("Triple Num: {}\n".format(triple_cnt))

    with open(e2idx_file, 'w', encoding="UTF-8") as f:
        for e_name in e2idx:
            f.write("{}\t{}\n".format(e2idx[e_name], e_name))
    with open(r2idx_file, 'w', encoding="UTF-8") as f:
        for r_name in r2idx:
            f.write("{}\t{}\n".format(r2idx[r_name], r_name))
    with open(triple2idx_file, 'w', encoding="UTF-8") as f:
        for t in triple_list:
            f.write("{}\t{}\t{}\n".format(t[0], t[1], t[2]))


if __name__ == "__main__":
    data2idx()
