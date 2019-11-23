# %% tach label voi text
import pandas as pd
# import re
# with open('C:/Users/Admin/Desktop/topic_detection_train.v1.0.txt', encoding="utf8") as f:
#     content = f.readlines()

import re
with open('C:/Users/Admin/Desktop/data.txt', encoding="utf8") as f:
    content = f.readlines()

regex = re.compile(r'^\S*')
result = regex.search(content[0])
print(result[0])

# %% tach label
label = []
for i in range(0, len(content)):
    topic = regex.search(content[i])
    label.append(topic[0])
    content[i] = content[i].replace(topic[0], '')

# %% bien label sang dang encoder
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(label)
total_label = le.transform(label)
print(len(set(label)))

# %%
df = pd.DataFrame(content, columns=['Essay'])
df['Label'] = total_label
# %%
labels = df['Label']
data = df['Essay']

from pyvi import ViTokenizer, ViPosTagger  # thư viện NLP tiếng Việt
from tqdm import tqdm
import numpy as np
import gensim  # thư viện NLP

VALIDATION_SPLIT = 0.2
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_test = data[-nb_validation_samples:]
y_test = labels[-nb_validation_samples:]

# %% Tokenizer
X_data = []
for x in x_train:
    essay = ViTokenizer.tokenize(x)
    X_data.append(essay)
X_test = []
for x in x_test:
    essay = ViTokenizer.tokenize(x)
    X_test.append(essay)

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# create a count vectorizer object
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(X_data)

# transform the training and validation data using count vectorizer object
X_data_count = count_vect.transform(X_data)
X_test_count = count_vect.transform(X_test)

# %% TF-IDF
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', max_features=500000, ngram_range=(1, 3))
tfidf_vect_ngram.fit(X_data)
X_data_tfidf_ngram = tfidf_vect_ngram.transform(X_data)
X_test_tfidf_ngram = tfidf_vect_ngram.transform(X_test)

# %% Naive Bayes
from sklearn.naive_bayes import MultinomialNB

model_bayes = MultinomialNB().fit(X_data_tfidf_ngram, y_train)
train_predictions_bayes = model_bayes.predict(X_data_tfidf_ngram)
from sklearn.metrics import accuracy_score

acc_bayes = accuracy_score(train_predictions_bayes, y_train)
print('Accuracy Bayes in train: ')
print(acc_bayes)

test_predictions_bayes = model_bayes.predict(X_test_tfidf_ngram)
acc_bayes_test = accuracy_score(test_predictions_bayes, y_test)
print('Accuracy Bayes in test: ')
print(acc_bayes_test)

# %% SVM
from sklearn import svm

model_SVC = svm.SVC().fit(X_data_tfidf_ngram, y_train)
train_predictions_bayes = model_SVC.predict(X_data_tfidf_ngram)
from sklearn.metrics import accuracy_score

acc_bayes = accuracy_score(train_predictions_bayes, y_train)
print('Accuracy SVM in train: ')
print(acc_bayes)
test_predictions_bayes = model_bayes.predict(X_test_tfidf_ngram)
acc_bayes_test = accuracy_score(test_predictions_bayes, y_test)
print('Accuracy SVM in test: ')
print(acc_bayes_test)
