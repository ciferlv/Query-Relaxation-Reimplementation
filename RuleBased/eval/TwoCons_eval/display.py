import numpy as np

r_name_list = ["birthPlace_award",
               "regionServed_location",
               "product_foundationPlace",
               "residence_deathPlace",
               "starring_birthPlace",
               "owner_foundationPlace",
               "regionServed_product",
               "locationCountry_foundationPlace",
               "regionServed_owner",
               "Average"]

metric_list = []
for r_name in r_name_list[:-1]:
    print(r_name)
    embed_metric = "./{}/metric.txt".format(r_name)
    cwa_metric = "./{}/cwa_rdqe_metric.txt".format(r_name)
    pca_metric = "./{}/pca_rdqe_metric.txt".format(r_name)
    embed_list = []
    cwa_list = []
    pca_list = []

    for line in open(embed_metric, 'r', encoding="UTF-8").readlines():
        token, value = line.split(":")
        value = round(float(value), 3)
        if token == "mrr":
            e_mrr = value
        if token == "hits10":
            e_hits10 = value
        if token == "average_time":
            e_time = value

    for line in open(cwa_metric, 'r', encoding="UTF-8").readlines():
        token, value = line.split("\t")
        value = round(float(value), 3)
        if token == "mrr":
            c_mrr = value
        if token == "hits10":
            c_hits10 = value
        if token == "average time":
            c_time = value

    for line in open(pca_metric, 'r', encoding="UTF-8").readlines():
        token, value = line.split("\t")
        value = round(float(value), 3)
        if token == "mrr":
            p_mrr = value
        if token == "hits10":
            p_hits10 = value
        if token == "average time":
            p_time = value

    metric_list.append([e_hits10, p_hits10, c_hits10, e_mrr, p_mrr, c_mrr, e_time, p_time, c_time])

t_res = list(np.mean(np.array(metric_list), axis=0))
t_res = [round(num, 3) for num in t_res]
metric_list.append(t_res)

res_str = ""
for r_name_idx, one_list in enumerate(metric_list):
    one_array = np.array(one_list).reshape([3, 3])
    one_res = ""
    for idx, metric_one in enumerate(one_array):
        if idx != 2:
            spc_idx = np.where(metric_one == np.max(metric_one))
        else:
            spc_idx = np.where(metric_one == np.min(metric_one))
        for m_idx, m in enumerate(metric_one):
            if m_idx in spc_idx[0]:
                one_res += "&\\textbf{" + str(m) + "}"
            else:
                one_res += "&" + str(m)
    if r_name_idx != len(metric_list) - 1:
        res_str += "{}{}\\\\\n\\hline\n".format("E{}".format(r_name_idx + 11), one_res)
    else:
        res_str += "{}{}\\\\\n\\hline\n".format("Average", one_res)
print(res_str)
