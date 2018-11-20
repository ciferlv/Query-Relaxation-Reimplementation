from RuleBased.Params import mydb, prefix_uri

folder = "F:\\Data\\dbpedia\\"
rdf_file = folder + "mappingbased_objects_en.ttl"

e2idx_file = folder + "e2idx.txt"
e2idx_shortcut_file = folder + "e2idx_shortcut.txt"

r2idx_file = folder + "r2idx.txt"
r2idx_shortcut_file = folder + "r2idx_shortcut.txt"

triple2idx_file = folder + "triple2idx.txt"
statistics_file = folder + "statistics.txt"


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
            if "inv_{}".format(r) not in r2idx:
                r2idx["inv_{}".format(r)] = r_cnt
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


def create_table():
    sql = """
    CREATE TABLE IF NOT EXISTS `fb15k`(
        `id` INT UNSIGNED AUTO_INCREMENT,
        `relation_idx` VARCHAR(200) NOT NULL,
        `rule_key` VARCHAR(200) NOT NULL,
        `rule_len` INT NOT NULL,
        `correct_ht` TEXT,
        `wrong_ht` TEXT,
        `no_idea_ht` TEXT,
        `P` FLOAT,
        `R` FLOAT,
        `F1` FLOAT,
        PRIMARY KEY ( `id` )
        )ENGINE=InnoDB DEFAULT CHARSET=utf8;
    """
    mycursor = mydb.cursor()
    try:
        mycursor.execute(sql)
        print("Success Creating table")
    except Exception as e:
        print(e)
    mydb.close()


'''
Change full name uri to its shortcut
Parameters:
-----------
uri: string
for example, <http://dbpedia.org/resource/Catalan_language>

Returns:
-----------
out: string
input uri's shortcut, for example, 'dbr:Catalan_language'
'''


def getShortcutName(uri):
    shortcut_uri = None
    uri = uri.strip().strip("<").strip(">")
    for key in prefix_uri.keys():
        if uri.startswith(key):
            length = len(key)
            shortcut_uri = "{}:{}".format(prefix_uri[key], uri[length:])
            break
    if shortcut_uri is None:
        return "<" + uri + ">"
    return shortcut_uri


'''
Shortcut every uri of e2idx.txt and r2idx.txt
'''


def shortCutURI():
    e_res_str = ""
    with open(e2idx_file, 'r', encoding="UTF-8") as f:
        for i, line in enumerate(f.readlines()):
            idx, name = line.strip().split()
            e_res_str += "{}\t{}\n".format(idx, getShortcutName(name))
    with open(e2idx_shortcut_file, 'w', encoding="UTF-8") as f:
        f.write(e_res_str.strip())

    r_res_str = ""
    with open(r2idx_file, 'r', encoding="UTF-8") as f:
        for line in f.readlines():
            idx, name = line.strip().split()
            r_res_str += "{}\t{}\n".format(idx, getShortcutName(name))
    with open(r2idx_shortcut_file, 'w', encoding="UTF-8") as f:
        f.write(r_res_str.strip())


if __name__ == "__main__":
    # data2idx()
    # create_table()
    # uri = "http://dbpedia.org/ontology/country"
    # print(getShortcutName(uri))
    shortCutURI()
