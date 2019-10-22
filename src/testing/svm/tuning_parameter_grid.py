import numpy as np

from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from src.preprocess import process

X_train, X_test, y_train, y_test = process('train')
print("Tuning...")
grid_param = {"C": np.arange(2, 10, 2),
             "gamma": np.arange(0.1, 1, 0.2)}

model = estimator = svm.SVC()
gd_sr = GridSearchCV(model, param_grid=grid_param, scoring='accuracy', cv=5, n_jobs=-1)
gd_sr.fit(X_train, y_train)
y_train_pred = classification_report(y_train, gd_sr.predict(X_train))
y_test_pred  = classification_report(y_test, gd_sr.predict(X_test))
print("""【{model_name}】\n Train Accuracy: \n{train}
           \n Test Accuracy:  \n{test}""".format(model_name=model.__class__.__name__, train=y_train_pred, test=y_test_pred))
best_parameters = gd_sr.best_params_
print(best_parameters)
