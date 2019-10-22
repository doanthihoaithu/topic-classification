from scipy import stats
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.model_selection import RandomizedSearchCV
from src.preprocess import process

X_train, X_test, y_train, y_test = process('train')
print("Tuning...")
random_param = {"C": stats.uniform(2, 10), "gamma": stats.uniform(0.1, 1)}

model = estimator = svm.SVC()
rd_sr = RandomizedSearchCV(model, param_distributions=random_param, scoring='accuracy', cv=5, n_jobs=-1)
rd_sr.fit(X_train, y_train)
y_train_pred = classification_report(y_train, rd_sr.predict(X_train))
y_test_pred  = classification_report(y_test, rd_sr.predict(X_test))
print("""【{model_name}】\n Train Accuracy: \n{train}
           \n Test Accuracy:  \n{test}""".format(model_name=model.__class__.__name__, train=y_train_pred, test=y_test_pred))
best_parameters = rd_sr.best_params_
print(best_parameters)
