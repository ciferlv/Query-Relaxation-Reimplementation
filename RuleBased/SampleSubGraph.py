import os

folder = "../../MyData/DBO/All/"
rdf_file = folder + "mappingbased_objects_en.ttl"

e2idx_file = folder + "e2idx.txt"
e2idx_shortcut_file = folder + "e2idx_shortcut.txt"

r2idx_file = folder + "r2idx.txt"
r2idx_shortcut_file = folder + "r2idx_shortcut.txt"

triple2idx_file = folder + "triple2idx.txt"
statistics_file = folder + "statistics.txt"

# def subGraph_mention_num(filtered_num):
#     filtered_by_num_folder = folder + "filtered_by_num" + file_path_seg
#     if not os.path.isdir(filtered_by_num_folder):
#         os.makedirs(filtered_by_num_folder)
#
#     filtered_triple2idx_file = filtered_by_num_folder + "filtered_triple2idx_" + str(
#         filtered_num) + ".txt"
#
#     e_num_dict = {}
#     r_num_dict = {}
#     triple2idx_list = []
#     with open(triple2idx_file, 'r', encoding="UTF-8") as f:
#         for line in f.readlines():
#             h, r, t = line.split()
#             if h not in e_num_dict:
#                 e_num_dict[h] = 0
#             if t not in e_num_dict:
#                 e_num_dict[t] = 0
#             if r not in r_num_dict:
#                 r_num_dict[r] = 0
#             e_num_dict[h] += 1
#             e_num_dict[t] += 1
#             r_num_dict[r] += 1
#             triple2idx_list.append([h, r, t])
#     e_saved_set = set()
#     for key in e_num_dict:
#         if e_num_dict[key] > filtered_num:
#             e_saved_set.add(key)
#
#     r_saved_set = set()
#     for key in r_num_dict:
#         if r_num_dict[key] > filtered_num:
#             r_saved_set.add(key)
#
#     saved_triple_num = 0
#     with open(filtered_triple2idx_file, 'w', encoding="UTF-8") as f:
#         for hrt in triple2idx_list:
#             h = hrt[0]
#             r = hrt[1]
#             t = hrt[2]
#             if (h in e_saved_set and t in e_saved_set) and int(r) != 0:
#                 f.write("{}\t{}\t{}\n".format(h, r, t))
#                 saved_triple_num += 1
#
#     print("Saved_E NUM: {}\t Saved_R NUM: {}\t Saved_Triple NUM: {}\n".format(
#         len(e_saved_set), len(r_saved_set), saved_triple_num))


def record_data_to_file(folder_name, idx2ename_dict, idx2rname_dict, triple_list):
    entity2id_file = folder_name + "/entity2id.txt"
    relation2id_file = folder_name + "/relation2id.txt"
    train2id_file = folder_name + "/triple2id.txt"
    new_statistics_file = folder_name + "statistics.txt"

    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)

    with open(train2id_file, 'w', encoding="UTF-8") as f:
        f.write("{}\n".format(len(triple_list)))
        for hrt in triple_list:
            f.write("{}\t{}\t{}\n".format(hrt[0], hrt[2],hrt[1]))

    with open(entity2id_file, 'w', encoding="UTF-8") as f:
        f.write("{}\n".format(len(idx2ename_dict)))
        for e_idx in idx2ename_dict:
            f.write("{}\t{}\n".format(idx2ename_dict[e_idx], e_idx))

    with open(relation2id_file, 'w', encoding="UTF-8") as f:
        f.write("{}\n".format(len(idx2rname_dict)))
        for r_idx in idx2rname_dict:
            f.write("{}\t{}\n".format(idx2rname_dict[r_idx], r_idx))

    with open(new_statistics_file, 'w', encoding="UTF-8") as f:
        f.write("e num:\t{}\n".format(len(idx2ename_dict)))
        f.write("r num:\t{}\n".format(len(idx2rname_dict)))
        f.write("triple num:\t{}\n".format(len(triple_list)))


def subGraph_country(country_name, folder_name):
    idx2e_dict = {}
    idx2r_dict = {}
    r2idx_dict = {}

    tar_country_idx = -1
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

    relabeled_ridx_dict = {}
    relabeled_eidx_dict = {}
    relabeled_triple_list = []

    relabeled_idx2rname_dict = {}
    relabeled_idx2ename_dict = {}

    r_cnt = 0
    e_cnt = 0

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

                if r_idx not in relabeled_ridx_dict:
                    relabeled_ridx_dict[r_idx] = r_cnt
                    relabeled_idx2rname_dict[r_cnt] = idx2r_dict[r_idx]
                    r_cnt += 1

                r_name = idx2r_dict[r_idx]
                inv_r_idx = r2idx_dict["inv_{}".format(r_name)]
                if inv_r_idx not in relabeled_ridx_dict:
                    relabeled_ridx_dict[inv_r_idx] = r_cnt
                    relabeled_idx2rname_dict[r_cnt] = "inv_{}".format(r_name)
                    r_cnt += 1

                if h_idx not in relabeled_eidx_dict:
                    relabeled_eidx_dict[h_idx] = e_cnt
                    relabeled_idx2ename_dict[e_cnt] = idx2e_dict[h_idx]
                    e_cnt += 1

                if t_idx not in relabeled_eidx_dict:
                    relabeled_eidx_dict[t_idx] = e_cnt
                    relabeled_idx2ename_dict[e_cnt] = idx2e_dict[t_idx]
                    e_cnt += 1

                relabeled_triple_list.append(
                    [relabeled_eidx_dict[h_idx],
                     relabeled_ridx_dict[r_idx],
                     relabeled_eidx_dict[t_idx]])

    relabeled_folder = "../../MyData/DBO/" + folder_name
    print(relabeled_folder)
    record_data_to_file(relabeled_folder, relabeled_idx2ename_dict, relabeled_idx2rname_dict, relabeled_triple_list)


if __name__ == "__main__":
    # subGraph_mention_num(100)
    country_name = "dbr:United_States"
    print(country_name.split(":")[-1])
    subGraph_country(country_name, "{}/".format(country_name.split(":")[-1]))
