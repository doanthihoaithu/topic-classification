from sklearn.linear_model import  SGDClassifier

class LinearModel(object):
    def __init__(self):
        self.clf = SGDClassifier()
