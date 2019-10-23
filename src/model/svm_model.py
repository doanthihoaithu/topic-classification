from sklearn.pipeline import Pipeline
# from sklearn.feature_extraction.text import TfidfTransformer
from src.preprocess import process
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.linear_model import  SGDClassifier

class SVMModel(object):
    def __init__(self):
        self.clf = SGDClassifier()
