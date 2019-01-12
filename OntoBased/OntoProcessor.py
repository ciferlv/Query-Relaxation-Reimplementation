from rdflib import Graph
import pprint
import queue

from Data.ReformData import getShortcutName


class SubBasics:
    def __init__(self, sub):
        self.sub = sub
        # father class set
        self.fc_set = set()
        # child class set
        self.cc_set = set()
        # father property set
        self.fp_set = set()
        # child property set
        self.cp_set = set()
        # domain set
        self.don_set = set()
        # range set
        self.rng_set = set()

    def add_fc(self, a_fc):
        self.fc_set.add(a_fc)

    def add_cc(self, a_cc):
        self.cc_set.add(a_cc)

    def add_fp(self, a_fp):
        self.fp_set.add(a_fp)

    def add_cp(self, a_cp):
        self.cp_set.add(a_cp)

    def add_don(self, a_don):
        self.don_set.add(a_don)

    def add_rng(self, a_rng):
        self.rng_set.add(a_rng)

    def add_info(self, _type, _uri):
        if _type == "rdfs:subPropertyOf":
            self.add_fp(_uri)
        if _type == "rdfs:subClassOf":
            self.add_fc(_uri)
        if _type == "inv_rdfs:subPropertyOf":
            self.add_cp(_uri)
        if _type == "inv_rdfs:subClassOf":
            self.add_cc(_uri)
        if _type.strip("inv_") == "rdfs:domain":
            self.add_don(_uri)
        if _type.strip("inv_") == "rdfs:range":
            self.add_rng(_uri)

    def __str__(self):
        sc_str = "\t".join(list(self.fc_set))
        sp_str = "\t".join(list(self.fp_set))
        don_str = "\t".join(list(self.don_set))
        rng_str = "\t".join(list(self.rng_set))
        return "sub:\t" + self.sub + "\n" \
               + "sc:\t" + sc_str + "\n" + "sp:\t" + sp_str + "\n" \
               + "don:\t" + don_str + "\n" + "rng:\t" + rng_str


class DbpediaOntology:
    def __init__(self, file_path_dict):
        self.file_path_dict = file_path_dict
        self.g = Graph()

        # sub: Object(SubBasics)
        self.sub_dict = {}

    def load_data(self):
        ontology_file = self.file_path_dict["folder"] + self.file_path_dict["ontology_file"]
        self.g.parse(ontology_file, format="nt")

        for sub, pred, obj in self.g:
            pred = getShortcutName(pred.n3())
            if pred in ["rdfs:subPropertyOf", "rdfs:subClassOf", "rdfs:domain", "rdfs:range"]:
                sub = getShortcutName(sub.n3())
                obj = getShortcutName(obj.n3())
                if sub not in self.sub_dict:
                    self.sub_dict[sub] = SubBasics(sub)
                self.sub_dict[sub].add_info(pred, obj)
                if obj not in self.sub_dict:
                    self.sub_dict[obj] = SubBasics(obj)
                self.sub_dict[obj].add_info("inv_" + pred, sub)
        # for subject in self.sub_dict:
        #     print(self.sub_dict[subject], end="\n\n")

    def gen_closure(self):
        print("Start generating closure.")
        zero_father_class_sub_list = []
        for subject in self.sub_dict:
            if len(self.sub_dict[subject].fc_set) == 0:
                zero_father_class_sub_list.append(subject)

        for sub in zero_father_class_sub_list:
            temp_queue = queue.Queue()
            temp_queue.put(sub)
            while not temp_queue.empty():
                top_sub = temp_queue.get()
                for child in self.sub_dict[top_sub].cc_set:
                    temp_queue.put(child)
                    self.sub_dict[child].fc_set |= self.sub_dict[top_sub].fc_set

        zero_father_property_sub_list = []
        for subject in self.sub_dict:
            if len(self.sub_dict[subject].fp_set) == 0:
                zero_father_property_sub_list.append(subject)

        for sub in zero_father_property_sub_list:
            temp_queue = queue.Queue()
            temp_queue.put(sub)
            while not temp_queue.empty():
                top_sub = temp_queue.get()
                for child in self.sub_dict[top_sub].cp_set:
                    temp_queue.put(child)
                    self.sub_dict[child].fp_set |= self.sub_dict[top_sub].fp_set
        print("Finish generating closure.")

    def deduction(self):
        print("Start deduction.")
        

def main():
    file_path_dict = {}
    with open("./file_path_store.txt", 'r', encoding="UTF-8") as f:
        for line in f.readlines():
            key_name, file_path = line.split()
            file_path_dict[key_name] = file_path

    do = DbpediaOntology(file_path_dict)
    do.load_data()


if __name__ == "__main__":
    main()
