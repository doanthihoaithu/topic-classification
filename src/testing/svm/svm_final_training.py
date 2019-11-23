import pickle

import pandas as pd
import re

from pyvi import ViTokenizer

from src.model.random_forest_model import RandomForestModel
from src.model.svm_model import SVMModel
from src.utils import get_project_root
ROOT_DIR = get_project_root()
print(ROOT_DIR)
data_file = str(ROOT_DIR) + "/resources/topic_detection_train.v1.0.txt"

with open(data_file, encoding="utf8") as f:
    content = f.readlines()
regex = re.compile(r'^\S*')
result = regex.search(content[0])
print(result[0])

label = []
for i in range(0, len(content)):
    topic = regex.search(content[i])
    label.append(topic[0])
    content[i] = content[i].replace(topic[0], '')

df = pd.DataFrame(content, columns=['Essay'])
df['Label'] = label
# print(df)

labels = df['Label']
# print(labels)
data = df['Essay']
# print(data)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, labels , test_size=0.20, random_state=1)
# print("Test data\n")
# print(X_test)

from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report

working_dir = str(ROOT_DIR) + "/src/testing/svm"
model_file_name = working_dir + "/" + "svm_final_model.sav"
def train(model):
     model.clf.fit(data,labels)
     pickle.dump(model, open(model_file_name, 'wb'))
     y_train_pred = classification_report(labels, model.clf.predict(data))

     print("""【{model_name}】\n Train Accuracy: \n{train}
           """.format(model_name=model.__class__.__name__, train=y_train_pred))
def check():
    loaded_model = pickle.load(open(model_file_name, 'rb'))
    predicts = loaded_model.clf.predict(data)
    y_train_pred = classification_report(labels, predicts)
    print("""【{model_name}】\n Train Accuracy: \n{train}
              """.format(model_name=loaded_model.__class__.__name__, train=y_train_pred))

train(SVMModel())
check()