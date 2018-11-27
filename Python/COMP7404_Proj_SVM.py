import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

print('##################### COMP7404 Project###########################')
print('#                                                               #')
print('#  Short Term Hang Seng Futures Price Prediction with SVM model #')
print('#                                                               #')
print('#################################################################')
print()

def runmodel(trainfile,testfile, col_X, col_Y):
    ###### Get training data ######
    traindata = pd.read_csv(os.getcwd()+ trainfile ) 
    
    ######split into training data X and Y by column names######
    train_X = traindata.loc[: ,col_X]
    train_Y_temp = traindata.loc[:,col_Y]

    ######Convery FLAT = 0, UP = 1, DOWN = 2######
    train_Y = np.zeros(train_Y_temp.shape[0])
    for i in range(train_Y_temp.shape[0]):
        if train_Y_temp[i] == 'FLAT':
            train_Y[i] = 0
        elif train_Y_temp[i] == 'UP':
            train_Y[i] = 1
        elif train_Y_temp[i] == 'DOWN':
            train_Y[i] = 2 

    ###### Get testing data ######    
    testdata = pd.read_csv(os.getcwd()+ testfile )
    
    ######split into training data X and Y by column names######
    test_X = testdata.loc[: , col_X]
    test_Y_temp = testdata.loc[:,col_Y]

    ######Convery FLAT = 0, UP = 1, DOWN = 2######
    test_Y = np.zeros(test_Y_temp.shape[0])
    for i in range(test_Y_temp.shape[0]):
        if test_Y_temp[i] == 'FLAT':
            test_Y[i] = 0
        elif test_Y_temp[i] == 'UP':    
            test_Y[i] = 1
        elif test_Y_temp[i] == 'DOWN':
            test_Y[i] = 2


    ###### Set parameters for cross-validation ######
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.1, 1, 10, 100],'C': [0.1, 1, 10, 100]},
                        {'kernel': ['linear'], 'C': [0.1, 1, 10, 100]}]
    
    scores = ['precision']
    
    ######split out the first 4000 sample for cross-validation######
    train_X_find = train_X[0:4000]
    train_Y_find = train_Y[0:4000]
    test_X_find = test_X[0:4000]
    test_Y_find = test_Y[0:4000]


    ###### Print cross-validation result######
    
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
            print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = test_Y_find, clf .predict(test_X_find)
        print(classification_report(y_true, y_pred))
        print()

        ###### Run full data with best parameters combination ######
        print("Run SVM with best parameters set:")

        svclassifier = SVC(**clf.best_params_)
        svclassifier.fit(train_X, np.ravel(train_Y))

        pred_y = svclassifier.predict(test_X)
        print("Test result:")
        print("Accuracy on testing data = " + str(svclassifier.score(test_X, test_Y)))
        print("Confusion Matrix:")
        print(confusion_matrix(test_Y,pred_y))
        print(classification_report(test_Y,pred_y))
        
        return

if __name__ == '__main__':
    print('#####For 1 min interval#####')
    #runmodel('/Test_1min.csv','/Train_1min.csv' ,['ROC5','Cl_to_SMA10','ROC10','ATR','MFI','OBV','ROC1','ADX','Vol.Change','TRIX.Sig'], 'Ret1Min')
    print()
    print()
    print()
    print('#####For 5 min interval#####')
    runmodel('/Test_5min.csv','/Train_5min.csv' ,['ROC5','TRIX.Sig','ATR','ROC10','OBV','MFI','ROC1','ADX','Cl_to_SMA60','Vol.Change'], 'Ret5Min')
    print()
    print()
    print()
    print('#####For 15 min interval#####')
    runmodel('/Test_15min.csv','/Train_15min.csv' ,['ROC5','Vol.Change','ATR','ROC10','OBV','MFI','TRIX','MACD','ADX','TRIX'], 'Ret15Min')
    print()
    print()
    print()
    print('#####For 30 min interval#####')
    runmodel('/Test_30min.csv','/Train_30min.csv' ,['ROC5','MACD','ROC10','ATR','MFI','OBV','TRIX.Sig','TRIX','Vol.Change','ADX'], 'Ret30Min')
    