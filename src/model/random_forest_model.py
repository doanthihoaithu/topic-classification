from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from src.transformer.feature_tranformer import FeatureTransformer
from sklearn.linear_model import SGDClassifier
from sklearn import svm


class RandomForestModel(object):
    def __init__(self):
        self.clf = self._init_pipeline()

    @staticmethod
    def _init_pipeline():
        pipe_line = Pipeline([
            ("transformer", FeatureTransformer()),
            ("vect", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            # ("clf-svm", SGDClassifier(loss='log', penalty='l2', alpha=1e-3, random_state=None))
            ("clf-rf", RandomForestClassifier(n_estimators=300, criterion="gini", bootstrap=True))
        ])

        return pipe_line