import pickle

import pandas as pd
import re

from pyvi import ViTokenizer

# with open('C:/Users/Admin/Desktop/topic_detection_train.v1.0.txt', encoding="utf8") as f:
#     content = f.readlines()
from src.model.random_forest_model import RandomForestModel

with open('C:/Users/Admin/Desktop/data.txt', encoding="utf8") as f:
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

def train(model):
     model.clf.fit(data,labels)
     model_file_name = 'random_forest_final_model.sav'
     pickle.dump(model, open(model_file_name, 'wb'))
     y_train_pred = classification_report(labels, model.clf.predict(data))

     print("""【{model_name}】\n Train Accuracy: \n{train}
           """.format(model_name=model.__class__.__name__, train=y_train_pred))
def check():
    model_file_name = 'random_forest_final_model.sav'
    loaded_model = pickle.load(open(model_file_name, 'rb'))
    y_train_pred = classification_report(y_test, loaded_model.clf.predict(X_test))
    print("""【{model_name}】\n Train Accuracy: \n{train}
              """.format(model_name=loaded_model.__class__.__name__, train=y_train_pred))

train(RandomForestModel())
check()