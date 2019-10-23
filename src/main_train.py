import pickle
import pandas as pd

from src.preprocess import process
#from src.model.svm_model import SVMModel
from src.model.linear_model import LinearModel
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test, x_test = process('train')
#model = SVMModel()
model = LinearModel()


def train(model):
    print("Runing...")
    model.clf.fit(X_train, y_train)
    # filename = 'train_svm_model_3_0.1.sav'
    # pickle.dump(model, open(filename, 'wb'))
    y_train_pred = classification_report(y_train, model.clf.predict(X_train))
    print("""【{model_name}】\n Train Accuracy: \n{train}
               """.format(model_name=model.__class__.__name__, train=y_train_pred))
    y_test_pred = classification_report(y_test, model.clf.predict(X_test))
    print("""【{model_name}】
                   \n Test Accuracy:  \n{test}""".format(model_name=model.__class__.__name__,
                                                         test=y_test_pred))
    test_predictions = model.clf.predict(X_test)
    acc_test = accuracy_score(test_predictions, y_test)
    print('Accuracy: ')
    print(acc_test)

    # wrong_data = []
    # right_y = []
    # wrong_predict = []
    # predicts = model.clf.predict(X_test)
    # print(predicts)
    # print(y_test.values)
    # for i in range(len(predicts)):
    #     if predicts[i] != y_test.values[i]:
    #         wrong_data.append(x_test.values[i])
    #         right_y.append(y_test.values[i])
    #         wrong_predict.append(predicts[i])
    # excell_frame = pd.DataFrame(wrong_data, columns=["Content"])
    # excell_frame["Right"] = right_y
    # excell_frame["Predict"] = wrong_predict
    # excell_file_name = 'test_fail_result.xlsx'
    # writer = pd.ExcelWriter(excell_file_name, engine='xlsxwriter')
    # excell_frame.to_excel(writer, sheet_name='Sheet1')
    # writer.save()
train(model)
