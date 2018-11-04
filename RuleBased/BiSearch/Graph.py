from RuleBased.BiSearch.Triple import Node


class Graph:
    def __init__(self, e2idx_file, r2idx_file, triple2idx_file):
        self.e2idx_file = e2idx_file
        self.r2idx_file = r2idx_file
        self.triple2idx_file = triple2idx_file

        self.e2idx = {}
        self.idx2e = {}
        self.r2idx = {}
        self.idx2r = {}
        self.node_dict = {}

    def load_data(self):
        with open(self.e2idx_file, 'r', encoding="UTF-8") as f:
            for line in f.readlines():
                idx, name = line.strip().split()
                self.e2idx[name] = int(idx)
                self.idx2e[int(idx)] = name
        with open(self.r2idx_file, 'r', encoding="UTF-8") as f:
            for line in f.readlines():
                idx, name = line.strip().split()
                self.r2idx[name] = int(idx)
                self.idx2r[idx] = name
        with open(self.triple2idx_file, 'r', encoding='UTF-8') as f:
            for line in f.readlines():
                h, r, t = line.strip().split()
                if int(h) not in self.node_dict:
                    self.node_dict[int(h)] = Node(h)
                if int(t) not in self.node_dict:
                    self.node_dict[int(t)] = Node(t)
                self.node_dict[int(h)].addPath(r=r, e=t)
                inv_r = "inv_{}".format(self.idx2r[r])
                self.node_dict[int(t)].addPath(r=self.r2idx[inv_r], e=h)

    def search_bidirect(self,head,tail,step):
        left = int(step/2)
        right = step - left

    def search_unidirect(self,head,step):
        res = []
        current_node_list = [[-1,head]]
        for i in range(step):
            temp_path = []
            for n in current_node_list:
                c_node = self.node_dict[n[1]]
                for path in c_node.path_list:
                    temp_n = n.copy()
                    temp_n.append(path.r)
                    temp_n.append(path.e)
                    temp_path.append(temp_n)
            res.append(temp_path.copy())
            current_node_list = temp_path.copy()
        return res





if __name__ == "__main__":
    e2idx_file = "./data/FB15K-237/e2idx.txt"
    r2idx_file = "./data/FB15K-237/r2idx.txt"
    triple2idx_file = "./data/FB15K-237/triple2idx.txt"
    graph = Graph(e2idx_file,r2idx_file,triple2idx_file)
    graph.load_data()
    res=graph.search_unidirect(2,2)
    for p in res:
        print(p)

