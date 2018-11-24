import os

folder = "F:\\Data\\dbpedia\\All"
rdf_file = folder + "mappingbased_objects_en.ttl"

e2idx_file = folder + "e2idx.txt"
e2idx_shortcut_file = folder + "e2idx_shortcut.txt"

r2idx_file = folder + "r2idx.txt"
r2idx_shortcut_file = folder + "r2idx_shortcut.txt"

triple2idx_file = folder + "triple2idx.txt"
statistics_file = folder + "statistics.txt"


def subGraph_mention_num(filtered_num):
    filtered_by_num_folder = folder + "filtered_by_num\\"
    if not os.path.isdir(filtered_by_num_folder):
        os.makedirs(filtered_by_num_folder)

    filtered_triple2idx_file = filtered_by_num_folder + "filtered_triple2idx_" + str(filtered_num) + ".txt"

    e_num_dict = {}
    r_num_dict = {}
    triple2idx_list = []
    with open(triple2idx_file, 'r', encoding="UTF-8") as f:
        for line in f.readlines():
            h, r, t = line.split()
            if h not in e_num_dict:
                e_num_dict[h] = 0
            if t not in e_num_dict:
                e_num_dict[t] = 0
            if r not in r_num_dict:
                r_num_dict[r] = 0
            e_num_dict[h] += 1
            e_num_dict[t] += 1
            r_num_dict[r] += 1
            triple2idx_list.append([h, r, t])
    e_saved_set = set()
    for key in e_num_dict:
        if e_num_dict[key] > filtered_num:
            e_saved_set.add(key)

    r_saved_set = set()
    for key in r_num_dict:
        if r_num_dict[key] > filtered_num:
            r_saved_set.add(key)

    saved_triple_num = 0
    with open(filtered_triple2idx_file, 'w', encoding="UTF-8") as f:
        for hrt in triple2idx_list:
            h = hrt[0]
            r = hrt[1]
            t = hrt[2]
            if (h in e_saved_set and
                t in e_saved_set) and int(r) != 0:
                f.write("{}\t{}\t{}\n".format(h, r, t))
                saved_triple_num += 1

    print("Saved_E NUM: {}\t Saved_R NUM: {}\t Saved_Triple NUM: {}\n".format(len(e_saved_set), len(r_saved_set),
                                                                              saved_triple_num))


def subGraph_country(country_name, folder_name):
    idx2e_dict = {}
    idx2r_dict = {}
    r2idx_dict = {}
    with open(e2idx_shortcut_file, 'r', encoding="UTF-8") as f:
        for line in f.readlines():
            e_idx, e_name = line.split()
            idx2e_dict[e_idx] = e_name
            if e_name == country_name:
                tar_country_idx = e_idx

    with open(r2idx_shortcut_file, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            r_idx, r_name = line.split()
            idx2r_dict[r_idx] = r_name
            r2idx_dict[r_name] = r_idx

    temp_eidx_set = set()
    filtered_ridx_set = set()
    filtered_eidx_set = set()
    filtered_triple_list = []
    with open(triple2idx_file, 'r', encoding="UTF-8") as f:
        for line in f.readlines():
            h_idx, r_idx, t_idx = line.split()
            if tar_country_idx == h_idx or t_idx == tar_country_idx:
                temp_eidx_set.add(h_idx)
                temp_eidx_set.add(t_idx)

    with open(triple2idx_file, 'r', encoding="UTF-8") as f:
        for line in f.readlines():
            h_idx, r_idx, t_idx = line.split()
            if h_idx in temp_eidx_set or t_idx in temp_eidx_set:
                filtered_triple_list.append([h_idx, r_idx, t_idx])
                filtered_ridx_set.add(r_idx)
                filtered_eidx_set.add(h_idx)
                filtered_eidx_set.add(t_idx)

    filtered_by_country_folder = folder + folder_name
    print(filtered_by_country_folder)

    if not os.path.isdir(filtered_by_country_folder):
        os.makedirs(filtered_by_country_folder)

    with open(filtered_by_country_folder + "triple2idx.txt", 'w', encoding="UTF-8") as f:
        for hrt in filtered_triple_list:
            f.write("{}\t{}\t{}\n".format(hrt[0], hrt[1], hrt[2]))

    with open(filtered_by_country_folder + "e2idx_shortcut.txt", 'w', encoding="UTF-8") as f:
        for e_idx in filtered_eidx_set:
            f.write("{}\t{}\n".format(e_idx, idx2e_dict[e_idx]))

    with open(filtered_by_country_folder + "r2idx_shortcut.txt", 'w', encoding="UTF-8") as f:
        for r_idx in filtered_ridx_set:
            r_name = idx2r_dict[r_idx]
            inv_r_name = "inv_{}".format(r_name)
            inv_r_idx = r2idx_dict[inv_r_name]
            f.write("{}\t{}\n".format(r_idx, idx2r_dict[r_idx]))
            f.write("{}\t{}\n".format(inv_r_idx, inv_r_name))

    with open(filtered_by_country_folder + "statistics.txt", 'w', encoding="UTF-8") as f:
        f.write("e num:\t{}\n".format(len(filtered_eidx_set)))
        f.write("r num:\t{}\n".format(len(filtered_ridx_set)))
        f.write("triple num:\t{}\n".format(len(filtered_triple_list)))


if __name__ == "__main__":
    # subGraph_mention_num(100)
    country_name = "dbr:Asia"
    # country_name = "dbr:Canada"
    # country_name = "dbr:United_States"
    print(country_name.split(":")[-1])
    subGraph_country(country_name, "{}\\".format(country_name.split(":")[-1]))


