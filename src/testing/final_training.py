# Tạo mô hình SVM cuối cùng train trên toàn bộ dữ liệu thầy gửi

import pickle

from sklearn.metrics import classification_report

from src.model.svm_model import SVMModel

import pandas as pd
import re

with open('topic_detection_train.v1.0.txt', encoding="utf8") as f:
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

#chia tập dữ liệu thành bộ train và bộ test - để có dữ liệu kiểm tra mô hình chạy được không
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data, labels , test_size=0.20, random_state=1)

model = SVMModel()
clf = model.clf.fit(data, labels)
filename = 'final_svm_model_3_0.1.sav'
pickle.dump(model, open(filename, 'wb'))
y_train_pred = classification_report(y_train, model.clf.predict(x_train))
print("""【{model_name}】\n Train Accuracy: \n{train}
          """.format(model_name=model.__class__.__name__, train=y_train_pred))