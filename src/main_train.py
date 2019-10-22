import pickle

from src.preprocess import process
from src.model.svm_model import SVMModel
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = process('train')
model = SVMModel()

def train(model):
    model.clf.fit(X_train, y_train)
    filename = 'train_svm_model_3_0.1.sav'
    pickle.dump(model, open(filename, 'wb'))
    y_train_pred = classification_report(y_train, model.clf.predict(X_train))
    print("""【{model_name}】\n Train Accuracy: \n{train}
               """.format(model_name=model.__class__.__name__, train=y_train_pred))
    y_test_pred = classification_report(y_test, model.clf.predict(X_test))
    print("""【{model_name}】
                   \n Test Accuracy:  \n{test}""".format(model_name=model.__class__.__name__,
                                                         test=y_test_pred))
train(model)
