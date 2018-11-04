import random


class Path:
    def __init__(self, r, e):
        self.r = int(r)
        self.e = int(e)

    def __eq__(self, other):
        return int(self.r) == int(other.r) and int(self.e) == int(other.e)


class Node:
    def __init__(self, e_key):
        self.e_key = int(e_key)
        self.path_list = []
        self.sampled_path_list = []

    def addPath(self, r, e):
        self.path_list.append(Path(r=r, e=e))

    def sample_path(self, threshold):
        if len(self.path_list) > threshold:
            self.sampled_path_list = random.sample(self.path_list, threshold)
        else:
            self.sampled_path_list = self.path_list.copy()

    def has_intersection(self, sampled_path):
        for one_p in sampled_path:
            if one_p in self.path_list:
                return True
        return False

    def gen_train_data(self):
        r_list = []
        e_list = []
        for one_p in self.sampled_path_list:
            r_list.append([int(one_p.r)])
            e_list.append([int(one_p.e)])
        return r_list, e_list


if __name__ == "__main__":
    a = Node('a')
    b = Node('b')
    a.addPath(r=2, e=7)
    a.addPath(r=5, e=9)
    b.addPath(r=2, e=3)
    b.addPath(r=5, e=0)
    b.addPath(r=7, e=3)
    print(a.has_intersection(b.path_list))
