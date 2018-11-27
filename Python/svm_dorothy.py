import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)


###### Get training data ######
traindata = pd.read_csv(os.getcwd()+'/Train_30min.csv' ) 
train_X = traindata.loc[: ,['ROC5','MACD','ROC10','ATR','MFI','OBV','TRIX.Sig','TRIX','Vol.Change','ADX']]
train_Y_temp = traindata.loc[:,"Ret30Min"]

train_Y = np.zeros(train_Y_temp.shape[0])
for i in range(train_Y_temp.shape[0]):
    if train_Y_temp[i] == 'FLAT':
        train_Y[i] = 0
    elif train_Y_temp[i] == 'UP':
        train_Y[i] = 1
    elif train_Y_temp[i] == 'DOWN':
        train_Y[i] = 2 

###### Get testing data ######    
testdata = pd.read_csv(os.getcwd()+'/Test_30min.csv' )
test_X = testdata.loc[: ,['ROC5','MACD','ROC10','ATR','MFI','OBV','TRIX.Sig','TRIX','Vol.Change','ADX']]
test_Y_temp = testdata.loc[:,"Ret30Min"]

test_Y = np.zeros(test_Y_temp.shape[0])
for i in range(test_Y_temp.shape[0]):
    if test_Y_temp[i] == 'FLAT':
        test_Y[i] = 0
    elif test_Y_temp[i] == 'UP':    
        test_Y[i] = 1
    elif test_Y_temp[i] == 'DOWN':
        test_Y[i] = 2

###### Set parameters for cross-validation ######
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.1, 1, 10, 100],
                     'C': [0.1, 1, 10, 100]},
                    {'kernel': ['linear'], 'C': [0.1, 1, 10, 100]}]
scores = ['precision']

train_X_find = train_X[0:3000]
train_Y_find = train_Y[0:3000]
test_X_find = test_X[0:3000]
test_Y_find = test_Y[0:3000]


###### Print cross-validation ######
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(), tuned_parameters, cv=5,scoring='%s_macro' % score)
    clf.fit(train_X_find, np.ravel(train_Y_find))

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = test_Y_find, clf .predict(test_X_find)
    print(classification_report(y_true, y_pred))
    print()


###### Run full test by best parameters combination ######
print("Running....")

svclassifier = SVC(C=100, kernel='rbf',gamma=1) ##(**clf.best_params_)
svclassifier.fit(train_X, np.ravel(train_Y))

pred_y = svclassifier.predict(test_X)
print("Test result:")
print("Accuracy on testing data = " + str(svclassifier.score(test_X, test_Y)))
print("Confusion Matrix:")
print(confusion_matrix(test_Y,pred_y))
print(classification_report(test_Y,pred_y))
