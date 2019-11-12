import pickle
import re
from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report


def get_project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent

def predict_and_save_to_txt(test_file, model_file, result_file):
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
    predicts = loaded_model.clf.predict(data)
    with open(result_file, 'w') as f:
        for item in predicts:
            f.write("%s\n" % item)
    y_test_pred = classification_report(labels,predicts )
    print("""【{model_name}】
       #           \n Test Accuracy:  \n{test}""".format(model_name=loaded_model.__class__.__name__,
                                                         test=y_test_pred))

def predict_and_save_to_xlsx(test_file, model_file, result_file):
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
    import pandas as pd
    df = pd.DataFrame(content, columns=['Essay'])
    df['Label'] = label
    # print(df)
    labels = df['Label']
    # print(labels)
    data = df['Essay']
    # print(data)

    loaded_model = pickle.load(open(model_file, 'rb'))
    predicts = loaded_model.clf.predict(data)
    import pandas as pd
    import xlsxwriter
    excell_frame = pd.DataFrame(predicts, columns=["Predicts"])
    excell_file_name = result_file
    writer = pd.ExcelWriter(excell_file_name, engine='xlsxwriter')
    excell_frame.to_excel(writer, sheet_name='Sheet1')
    writer.save()
    y_test_pred = classification_report(labels,predicts )
    print("""【{model_name}】
       #           \n Test Accuracy:  \n{test}""".format(model_name=loaded_model.__class__.__name__,
                                                         test=y_test_pred))