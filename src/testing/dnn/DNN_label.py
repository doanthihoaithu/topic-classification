# -*- coding: utf-8 -*-
from sklearn.datasets import fetch_20newsgroups
from keras.layers import Dropout, Dense
from keras.models import Sequential
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn import metrics
import pandas as pd
import re
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from pyvi.ViTokenizer import ViTokenizer
from pyvi.ViPosTagger import ViPosTagger
from sklearn.feature_extraction.text import CountVectorizer

import pickle

from src.model.DNN_model import DNNModel

SPECIAL_CHARACTER = '0123456789%@$.,=+-!;/()*"&^:#|\n\t\''


def dump_to_file(obj, path):
    pickle.dump(obj, open(path, 'wb'))


def load_obj_from_file(path):
    obj = pickle.load(open(path, 'rb'))
    return obj


# with open('src/testing/data.txt', encoding="utf8") as f:
with open('topic_detection_train.v1.0.txt', encoding="utf8") as f:
    content = f.readlines()
regex = re.compile(r'^\S*')
# result = regex.search(content[0])
# print(result)
# print(content[0])
# print(ViTokenizer.tokenize(content[0]))

# content = content[:1000]
# content = content[:1000]
# content = content[2000:3000]
print(len(content))

label = []
for i in range(0, len(content)):
    topic = regex.search(content[i])
    label.append(topic[0])
    temp_str = content[i].replace(topic[0], '')
    temp_str = ' '.join([x.strip(SPECIAL_CHARACTER).lower() for x in temp_str.split()])
    temp_str = ViTokenizer.tokenize(temp_str)
    content[i] = temp_str
    # content[i] = content[i].replace(topic[0], '')
    # content[i] = ViTokenizer.tokenize(content[i])

# print(label)
encoder = LabelEncoder()
label = encoder.fit_transform(label)
# print(label)

dump_to_file(encoder, "label_encoder.pk")

encoder.inverse_transform()