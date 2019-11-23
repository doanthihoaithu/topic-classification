import pickle
import re
import pandas as pd
from sklearn.metrics import classification_report


def check(test_file, model_file):
    with open(test_file, encoding="utf8") as f:
        content = f.readlines()

    regex = re.compile(r'^\S*')
    result = regex.search(content[0])
    print(result[0])

    # tách label
    label = []
    for i in range(0, len(content)):
        topic = regex.search(content[i])
        label.append(topic[0])
        content[i] = content[i].replace(topic[0], '')

    # đổ dữ liệu vào data frame
    df = pd.DataFrame(content, columns=['Essay'])
    df['Label'] = label
    # print(df)
    labels = df['Label']
    # print(labels)
    data = df['Essay']
    # print(data)

    loaded_model = pickle.load(open(model_file, 'rb'))
    y_test_pred = classification_report(labels, loaded_model.clf.predict(data))
    print("""【{model_name}】
       #           \n Test Accuracy:  \n{test}""".format(model_name=loaded_model.__class__.__name__,
                                                         test=y_test_pred))


def check_and_export_wrong_to_excell(test_file, model_file):
    with open(test_file, encoding="utf8") as f:
        content = f.readlines()

    regex = re.compile(r'^\S*')
    result = regex.search(content[0])
    print(result[0])

    # tách label
    label = []
    for i in range(0, len(content)):
        topic = regex.search(content[i])
        label.append(topic[0])
        content[i] = content[i].replace(topic[0], '')

    import pandas as pd
    # đổ dữ liệu vào data frame
    df = pd.DataFrame(content, columns=['Essay'])
    df['Label'] = label
    # print(df)
    labels = df['Label']
    # print(labels)
    data = df['Essay']
    # print(data)

    loaded_model = pickle.load(open(model_file, 'rb'))
    predicts = loaded_model.clf.predict(data)
    y_test_pred = classification_report(labels, predicts)
    print("""【{model_name}】
       #           \n Test Accuracy:  \n{test}""".format(model_name=loaded_model.__class__.__name__,
                                                         test=y_test_pred))

    y_test_list = labels.values
    x_test_list = data.values
    wrong_data = []
    wrong_y = []
    wrong_predict = []
    for i in range(len(predicts)):
        if predicts[i] != y_test_list[i]:
            wrong_data.append(x_test_list[i])
            wrong_y.append(y_test_list[i])
            wrong_predict.append(predicts[i])

    import pandas as pd
    import xlsxwriter
    excell_frame = pd.DataFrame(wrong_data, columns=["Content"])
    excell_frame["Result"] = wrong_y
    excell_frame["Predict"] = wrong_predict
    excell_file_name = test_file + '_test_fail_result.xlsx'
    writer = pd.ExcelWriter(excell_file_name, engine='xlsxwriter')
    excell_frame.to_excel(writer, sheet_name='Sheet1')
    writer.save()


test_file = "data.txt"
# test_file = "topic_detection_train.v1.0.txt"
model_file = "final_svm_model_3_0.1.sav"
# check(test_file, model_file)
check_and_export_wrong_to_excell(test_file, model_file)
