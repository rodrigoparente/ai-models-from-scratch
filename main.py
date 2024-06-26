# python imports
import warnings

# third-party imports
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import fetch_california_housing

# project imports
from classifiers import CustomLinearRegression
from classifiers import CustomLogisticRegression
from classifiers import CustomMLPClassifier
from commons.model_selection import train_test_split
from commons.preprocessing import one_hot_encode
from commons.preprocessing import standardized
from commons.utils import shuffle
from commons.utils import display_cm

# supress warnings
warnings.filterwarnings('ignore')


def execute_linear():
    # data preprocessing

    print('loading california housing dataset...\n')

    X, y = fetch_california_housing(return_X_y=True)

    X, y = shuffle(X, y)

    X = standardized(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # training custom model

    model = CustomLinearRegression()
    model.fit(X_train, y_train)

    y_predict = model.predict(X_test)

    print('## Custom Model\n')
    print(f'r2: {r2_score(y_test, y_predict):.4f}')
    print(f'rmse: {root_mean_squared_error(y_test, y_predict):.4f}')

    # training scikit model

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_predict = model.predict(X_test)

    print('\n## Scikit Learn Model\n')
    print(f'r2: {r2_score(y_test, y_predict):.4f}')
    print(f'rmse: {root_mean_squared_error(y_test, y_predict):.4f}')


def execute_logistic():
    # data preprocessing

    print('loading breast cancer dataset...\n')

    X, y = load_breast_cancer(return_X_y=True)

    X, y = shuffle(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # training custom model

    model = CustomLogisticRegression()
    model.fit(X_train, y_train)

    predicted_y = model.predict(X_test)

    print('## Custom Model\n')
    print(f'accuracy:  {accuracy_score(y_test, predicted_y):.4f}')
    print(f'precision: {precision_score(y_test, predicted_y):.4f}')
    print(f'recall:    {recall_score(y_test, predicted_y):.4f}')
    print(f'f1_score:  {f1_score(y_test, predicted_y):.4f}')
    print('confusion matrix:')
    display_cm(confusion_matrix(y_test, predicted_y, normalize='true'))

    # training scikit model

    model = LogisticRegression()
    model.fit(X_train, y_train)

    predicted_y = model.predict(X_test)

    print('\n## Scikit Learn Model\n')
    print(f'accuracy:  {accuracy_score(y_test, predicted_y):.4f}')
    print(f'precision: {precision_score(y_test, predicted_y):.4f}')
    print(f'recall:    {recall_score(y_test, predicted_y):.4f}')
    print(f'f1_score:  {f1_score(y_test, predicted_y):.4f}')
    print('confusion matrix:')
    display_cm(confusion_matrix(y_test, predicted_y, normalize='true'))


def execute_mlp():
    # data preprocessing

    print('loading iris dataset...\n')

    X, y = load_iris(return_X_y=True)

    X, y = shuffle(X, y)

    X = standardized(X)

    y = one_hot_encode(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # training custom model

    model = CustomMLPClassifier()
    model.fit(X_train, y_train)

    predicted_y = model.predict(X_test)
    y_test = np.argmax(y_test, axis=1)

    print('## Custom Model\n')
    print(f'accuracy:  {accuracy_score(y_test, predicted_y):.4f}')
    print(f'precision: {precision_score(y_test, predicted_y, average="weighted"):.4f}')
    print(f'recall:    {recall_score(y_test, predicted_y, average="weighted"):.4f}')
    print(f'f1_score:  {f1_score(y_test, predicted_y, average="weighted"):.4f}')
    print('confusion matrix:')
    display_cm(confusion_matrix(y_test, predicted_y, normalize='true'))

    # training scikit model

    model = MLPClassifier()
    model.fit(X_train, y_train)

    predicted_y = model.predict(X_test)
    predicted_y = np.argmax(predicted_y, axis=1)

    print('\n## Scikit Learn Model\n')
    print(f'accuracy:  {accuracy_score(y_test, predicted_y):.4f}')
    print(f'precision: {precision_score(y_test, predicted_y, average="weighted"):.4f}')
    print(f'recall:    {recall_score(y_test, predicted_y, average="weighted"):.4f}')
    print(f'f1_score:  {f1_score(y_test, predicted_y, average="weighted"):.4f}')
    print('confusion matrix:')
    display_cm(confusion_matrix(y_test, predicted_y, normalize='true'))


if __name__ == '__main__':

    print('# Linear Regression')
    print('-------------------\n')

    execute_linear()

    print('\n# Logistic Regression')
    print('---------------------\n')

    execute_logistic()

    print('\n# MLP Classifier')
    print('----------------\n')

    execute_mlp()
