import pickle

from src.preprocess import process
from src.model.svm_model import SVMModel
from sklearn.metrics import classification_report

X_data = process('test')
model = SVMModel()

def test():
    filename = 'train_svm_model_3_0.1.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    y_train_pred = classification_report(y_data, loaded_model.clf.predict(X_data))
    print("""【{model_name}】
                     \n Train Accuracy:  \n{train}""".format(model_name=loaded_model.__class__.__name__,
                                                             train=y_train_pred))
    # y_test_pred = classification_report(y_test, loaded_model.clf.predict(X_test))
    # print("""【{model_name}】
    #               \n Test Accuracy:  \n{test}""".format(model_name=loaded_model.__class__.__name__,
    #                                                     test=y_test_pred))
test()