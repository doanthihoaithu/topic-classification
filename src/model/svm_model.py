from sklearn.pipeline import Pipeline
# from sklearn.feature_extraction.text import TfidfTransformer
from src.preprocess import process
from sklearn.linear_model import SGDClassifier
from sklearn import svm


class SVMModel(object):
    def __init__(self):
        self.clf = svm.SVC(C=3, gamma=0.1)