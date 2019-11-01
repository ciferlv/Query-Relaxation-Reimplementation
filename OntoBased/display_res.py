with open("./onto_result_query.txt", 'r', encoding="UTF-8") as f:
    for idx, line in enumerate(f.read().split("[+=]")):
        new_file_path = "./display_res/E{}.txt".format(idx+1)
        with open(new_file_path, "w", encoding="UTF-8") as f:
            f.write(line)
