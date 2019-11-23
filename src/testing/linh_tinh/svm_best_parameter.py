import pickle

import pandas as pd
import re

from pyvi import ViTokenizer
from scipy import stats

# with open('data.txt', encoding="utf8") as f:
#     content = f.readlines()
from src.model.svm_model import SVMModel

with open('topic_detection_train.v1.0.txt', encoding="utf8") as f:
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
# df['Label'] = total_label
df['Label'] = label
# print(df)

labels = df['Label']
# print(labels)
data = df['Essay']
# print(data)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data, labels , test_size=0.20, random_state=1)

from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report

def train(model):
     model.clf.fit(x_train,y_train)
     filename = 'train_svm_model_3_0.1.sav'
     pickle.dump(model, open(filename, 'wb'))
     y_train_pred = classification_report(y_train,model.clf.predict(x_train))
     print("""【{model_name}】\n Train Accuracy: \n{train}
           """.format(model_name=model.__class__.__name__, train=y_train_pred))
     y_test_pred  = classification_report(y_test,model.clf.predict(x_test))
     print("""【{model_name}】
               \n Test Accuracy:  \n{test}""".format(model_name=model.__class__.__name__,
                                                     test=y_test_pred))
def check():
    filename = 'train_svm_model_3_0.1.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    y_train_pred = classification_report(y_train, loaded_model.clf.predict(x_train))
    print("""【{model_name}】
                 \n Train Accuracy:  \n{train}""".format(model_name=loaded_model.__class__.__name__,
                                                       train=y_train_pred))
    y_test_pred = classification_report(y_test, loaded_model.clf.predict(x_test))
    print("""【{model_name}】
              \n Test Accuracy:  \n{test}""".format(model_name=loaded_model.__class__.__name__,
                                                    test=y_test_pred))

model = SVMModel()

# gọi hàm này để train trên 80% dư liệu
# train(model)

# gọi hàm này để kiểm tra mô hình với 20% dữ liệu còn lại
check()
