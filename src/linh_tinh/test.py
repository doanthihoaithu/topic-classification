import pandas as pd
import re

from pyvi import ViTokenizer

with open('C:/Users/Admin/Desktop/topic_detection_train.v1.0.txt', encoding="utf8") as f:
    content = f.readlines()

regex = re.compile(r'^\S*')
result = regex.search(content[0])
print(result[0])

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

dftest = pd.DataFrame(set(label))
# print(dftest)
# df = pd.DataFrame(set(label))
# print(df)

df = pd.DataFrame(content, columns=['Essay'])
df['Label'] = total_label
# print(df)

labels = df['Label']
# print(labels)
data = df['Essay']
# print(data)

VALIDATION_SPLIT = 0.2
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_test = data[-nb_validation_samples:]
y_test = labels[-nb_validation_samples:]
# print("Test data")
# print(x_test)
# print(y_test)

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
# print(count_vect.get_feature_names())

# transform the training and validation data using count vectorizer object
X_data_count = count_vect.transform(X_data)
X_test_count = count_vect.transform(X_test)
# print(X_test_count.shape)

# %% TF-IDF
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', max_features=500000, ngram_range=(1, 3))
tfidf_vect_ngram.fit(X_data)
X_data_tfidf_ngram = tfidf_vect_ngram.transform(X_data)
X_test_tfidf_ngram = tfidf_vect_ngram.transform(X_test)
# print(X_test_tfidf_ngram.toarray())

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(data, labels , test_size=0.20, random_state=1)
# print("Test data\n")
# print(X_test)

from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report

def model_check(model):
     model.fit(X_data_tfidf_ngram,y_train)
     y_train_pred = classification_report(y_train,model.predict(X_data_tfidf_ngram))
     y_test_pred  = classification_report(y_test,model.predict(X_test_tfidf_ngram))

     print("""【{model_name}】\n Train Accuracy: \n{train}
           \n Test Accuracy:  \n{test}""".format(model_name=model.__class__.__name__, train=y_train_pred, test=y_test_pred))

from sklearn import svm

# model_svm = svm.SVC()
# print(model_check(model_svm))
print(model_check(RandomForestClassifier()))

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint as sp_randint


# grid_param = {
#     'n_estimators': [100, 300, 500, 800, 1000],
#     'criterion': ['gini', 'entropy'],
#     'bootstrap': [True, False]
# }

grid_param = {"max_depth": [3, None],                  #distribution
              "n_estimators":[50,100,200,300,400,500],
              "max_features": sp_randint(1, 11),
              "min_samples_split": sp_randint(2, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}


# gd_sr = GridSearchCV(estimator=RandomForestClassifier(), param_grid=grid_param, scoring='accuracy', cv=5, n_jobs=-1)
gd_sr = RandomizedSearchCV( estimator=RandomForestClassifier( random_state=0 ),
                                    param_distributions=grid_param,
                                    cv=3,              #CV
                                    n_iter=1944,          #interation num
                                    scoring="accuracy", #metrics
                                    n_jobs=3,           #num of core
                                    verbose=0,
                                    random_state=1)
gd_sr.fit(X_data_tfidf_ngram,y_train)
best_parameters = gd_sr.best_params_
print(best_parameters)
best_result = gd_sr.best_score_
print(best_result)