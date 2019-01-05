import http.server
import socketserver


class SimiGraph:
    def __init__(self, triple2idx_file):
        self.triple2idx_file = triple2idx_file

        self.U_e_w_num = {}
        self.U_e_num = {}
        self.B_e_w_num = {}
        self.B_e_num = {}
        self.U_w_num = {}
        self.e_B_w_num = {}

        self.S_r_w_num = {}
        self.S_r_num = {}
        self.O_r_w_num = {}
        self.O_r_num = {}
        self.B_r_w_num = {}
        self.B_r_num = {}

        self.r_B_w_num = {}
        self.S_w_num = {}
        self.O_w_num = {}

        self.total_e_num = 0
        self.total_s_o_num = 0
        self.total_e_r_num = 0
        self.total_s_num = 0
        self.total_o_num = 0

    # w_token is e,r
    def count_w_of_e_B(self, w_token):
        if w_token not in self.e_B_w_num:
            self.e_B_w_num[w_token] = 0
        self.e_B_w_num[w_token] += 1

    # key_e is e_id, doc_e is w_id（e_id）
    def add_U_e_w(self, key_e, doc_e):
        assert type(key_e) == type(1), "key_e has wrong type."
        assert type(doc_e) == type(1), "doc_e has wrong type."

        if key_e not in self.U_e_w_num:
            self.U_e_w_num[key_e] = {}
        if doc_e not in self.U_e_w_num[key_e]:
            self.U_e_w_num[key_e][doc_e] = 0
        self.U_e_w_num[key_e][doc_e] += 1

        if key_e not in self.U_e_num:
            self.U_e_num[key_e] = 0
        self.U_e_num[key_e] += 1

    # key_e is e_id, doc_e_r is w_token
    def add_B_e_w(self, key_e, doc_e_r):
        assert type(key_e) == type(1), "key_e has wrong type."
        assert type(doc_e_r) == type("1"), "doc_e has wrong type."
        if key_e not in self.B_e_w_num:
            self.B_e_w_num[key_e] = {}
        if doc_e_r not in self.B_e_w_num[key_e]:
            self.B_e_w_num[key_e][doc_e_r] = 0
        self.B_e_w_num[key_e][doc_e_r] += 1

        if key_e not in self.B_e_num:
            self.B_e_num[key_e] = 0
        self.B_e_num[key_e] += 1

    # key_r is r_id, w_id is subject of r_id
    def add_S_r_w(self, key_r, w_id):
        assert type(key_r) == type(1), "key_r has wrong type."
        assert type(w_id) == type(1), "e_subject has wrong type."
        if key_r not in self.S_r_w_num:
            self.S_r_w_num[key_r] = {}
        if w_id not in self.S_r_w_num[key_r]:
            self.S_r_w_num[key_r][w_id] = 0
        self.S_r_w_num[key_r][w_id] += 1

        if key_r not in self.S_r_num:
            self.S_r_num[key_r] = 0
        self.S_r_num[key_r] += 1

    # key_r is r_id, w_id is object of r_id
    def add_O_r_w(self, key_r, w_id):
        assert type(key_r) == type(1), "key_r has wrong type."
        assert type(w_id) == type(1), "e_object has wrong type."
        if key_r not in self.O_r_w_num:
            self.O_r_w_num[key_r] = {}
        if w_id not in self.O_r_w_num[key_r]:
            self.O_r_w_num[key_r][w_id] = 0
        self.O_r_w_num[key_r][w_id] += 1

        if key_r not in self.O_r_num:
            self.O_r_num[key_r] = 0
        self.O_r_num[key_r] += 1

    # key_r is r_id, w_token is 'subject_id,object_id' of r_id
    def add_B_r_w(self, key_r, w_token):
        assert type(key_r) == type(1), "key_r has wrong type."
        assert type(w_token) == type("1"), "o_s has wrong type."
        if key_r not in self.B_r_w_num:
            self.B_r_w_num[key_r] = {}
        if w_token not in self.B_r_w_num[key_r]:
            self.B_r_w_num[key_r][w_token] = 0
        self.B_r_w_num[key_r][w_token] += 1

        if key_r not in self.B_r_num:
            self.B_r_num[key_r] = 0
        self.B_r_num[key_r] += 1

    # w_id is the id of e as subject or object
    def count_U_w_num(self, w_id):
        if w_id not in self.U_w_num:
            self.U_w_num[w_id] = 0
        self.U_w_num[w_id] += 11

    def load_data(self):
        print("Start loading data.")
        with open(self.triple2idx_file, 'r', encoding="UTF-8") as f:
            for line in f.readlines():
                h, r, t = [int(w) for w in line.split()]
                self.total_e_num += 2
                self.total_e_r_num += 2
                self.total_s_o_num += 1
                self.total_s_num += 1
                self.total_o_num += 1

                self.count_U_w_num(h)
                self.count_U_w_num(t)
                self.count_w_of_e_B("{},{}".format(t, r))
                self.count_w_of_e_B("{},{}".format(h, r))

                self.add_U_e_w(h, t)
                self.add_U_e_w(t, h)
                self.add_B_e_w(h, "{},{}".format(t, r))
                self.add_B_e_w(t, "{},{}".format(h, r))
                self.add_S_r_w(r, h)
                self.add_O_r_w(r, t)
                self.add_B_r_w(r, "{},{}".format(h, t))

    def calculate_LM(self):
        print("Start Calculating LM")

        def change_num2ratio(every_num_dict, total_num_dict):
            for key_id in every_num_dict:
                for w_id in every_num_dict[key_id]:
                    every_num_dict[key_id][w_id] = every_num_dict[key_id][w_id] / total_num_dict[key_id]

        change_num2ratio(self.U_e_w_num, self.U_e_num)
        change_num2ratio(self.B_e_w_num, self.B_e_num)

        change_num2ratio(self.S_r_w_num, self.S_r_num)
        change_num2ratio(self.O_r_w_num, self.O_r_num)
        change_num2ratio(self.B_r_w_num, self.B_r_num)

        def change_num2ratio_1(every_num_dict, total_num):
            for w_id in every_num_dict:
                every_num_dict[w_id] = every_num_dict[w_id] / total_num

        change_num2ratio_1(self.U_w_num, self.total_e_num)
        change_num2ratio_1(self.e_B_w_num, self.total_e_r_num)
        change_num2ratio_1(self.S_w_num, self.total_s_num)
        change_num2ratio_1(self.O_w_num, self.total_o_num)
        change_num2ratio_1(self.r_B_w_num, self.total_s_o_num)

    def get_name_simi(self, entity_name):
        print("Get entity name: {}".format(entity_name))

    # def get_KL_(self,source_dict,target_dict):
    #


def main():
    folder = "F:\\Data\\dbpedia\\All\\"

    e2idx_file = folder + "e2idx.txt"
    e2idx_shortcut_file = folder + "e2idx_shortcut.txt"

    r2idx_file = folder + "r2idx.txt"
    r2idx_shortcut_file = folder + "r2idx_shortcut.txt"

    triple2idx_file = folder + "triple2idx.txt"

    simiGraph = SimiGraph(triple2idx_file)
    simiGraph.load_data()
    simiGraph.calculate_LM()

    while True:
        my_input = input("Please input: ")
        if my_input == "1":
            simiGraph.get_name_simi(my_input)


if __name__ == "__main__":
    main()
