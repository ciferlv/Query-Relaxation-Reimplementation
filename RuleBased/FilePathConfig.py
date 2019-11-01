import os

from RuleBased.Params import file_path_seg


class FilePathConfig:
    def __init__(self):
        # self.root_folder = "../../MyData/dbo_pca_plain/"
        # self.root_folder = "../../MyData/{}/".format(train_dataset_folder)
        self.root_folder = "../../MyData/{}/".format("DBO")
        self.train_scope = "United_States"
        self.test_scope = "Canada"
        self.search_scope = "All"

        self.all_idx2r_file = self.root_folder + "All" + file_path_seg + "r2idx_shortcut.txt"

        self.embed_vector_folder = "../../MyData/EmbedDBO/"

        self.search_root = self.root_folder + self.search_scope + file_path_seg
        self.train_root = self.root_folder + self.train_scope + file_path_seg
        self.test_root = self.root_folder + self.test_scope + file_path_seg

        self.trained_model_saved_folder = self.train_root + "model" + file_path_seg
        self.train_e2idx_file = self.train_root + "entity2id.txt"
        self.train_r2idx_file = self.train_root + "relation2id.txt"
        self.train_triple2idx_file = self.train_root + "triple2id.txt"

        self.test_e2idx_file = self.test_root + "e2idx_shortcut.txt"
        self.test_r2idx_file = self.test_root + "r2idx_shortcut.txt"
        self.test_triple2idx_file = self.test_root + "triple2idx.txt"

        self.search_e2idx_file = self.search_root + "e2idx_shortcut.txt"
        self.search_r2idx_file = self.search_root + "r2idx_shortcut.txt"
        self.search_triple2idx_file = self.search_root + "triple2idx.txt"

        self.e2vec_file = "./MyData/EmbedDBO/entity2vector.txt"
        self.r2vec_file = "./MyData/EmbedDBO/relation2vector.txt"

        self.embedding_result_all_folder = "./results/embeddingResult/all/"
        self.pra_result_all_folder = "./results/praResult/all/"
        self.embedding_result_top_K_folder = "./results/embeddingResult/topk/"
        self.pra_result_top_K_folder = "./results/praResult/topk/"

        self.check_folder()

        self.embedding_result_all_file_path = ""
        self.embedding_result_top_K_file_path = ""
        self.pra_result_all_file_path = ""
        self.pra_result_top_K_file_path = ""

    def createFolder(self, folder_name):
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

    def check_folder(self):
        self.createFolder(self.embedding_result_all_folder)
        self.createFolder(self.embedding_result_top_K_folder)
        self.createFolder(self.pra_result_all_folder)
        self.createFolder(self.pra_result_top_K_folder)

    def setEmbeddingResFile(self, idx):
        self.embedding_result_all_file_path = self.embedding_result_all_folder + "rule_based_all_embedding_{}.txt".format(
            idx + 1)
        self.embedding_result_top_K_file_path = self.embedding_result_top_K_folder + "rule_based_topK_embedding_{}.txt".format(
            idx + 1)

        self.pra_result_all_file_path = self.pra_result_all_folder + "rule_based_all_pra_{}.txt".format(idx + 1)
        self.pra_result_top_K_file_path = self.pra_result_top_K_folder + "rule_based_topK_pra_{}.txt".format(
            idx + 1)

        if os.path.exists(self.embedding_result_top_K_file_path) or os.path.exists(self.pra_result_top_K_file_path):
            return True

        return False
