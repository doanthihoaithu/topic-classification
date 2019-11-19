dnn_result = "result/dnn_result.txt"
random_forest_result = "result/random_forest_result.txt"
svm_result = "result/svm_result.txt"


with open(dnn_result, encoding="utf8") as f:
    dnn_result = f.readlines()

with open(random_forest_result, encoding="utf8") as f:
    random_forest_result = f.readlines()

with open(svm_result, encoding="utf8") as f:
    svm_result = f.readlines()

print(f"{len(dnn_result)} {len(random_forest_result)} {len(svm_result)}")


def compare_result(name_1, list_1, name_2, list_2, **kwargs):
    length = len(list_1)
    same_count = 0
    for i in range(length):
        if list_1[i] == list_2[i]:
            same_count = same_count + 1
    print(f"{name_1} vs {name_2}: {same_count/length}")


compare_result("DNN", dnn_result, "random_forest", random_forest_result)
compare_result("svm", svm_result, "random_forest", random_forest_result)
compare_result("DNN", dnn_result, "svm", svm_result)