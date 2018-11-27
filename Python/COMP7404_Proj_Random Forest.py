"""
Random Forest Prediction Model
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import timeit


######Get train data######
data1_train = pd.read_csv('D:\OneDrive\Course\COMP7404\Project\Data\Train_1min.csv',
                   index_col=0, header=0)

Y1_train = np.ravel(data1_train['Ret1Min'])       
X1_train = data1_train.loc[:,['ROC5', 'Cl_to_SMA10', 'ATR', 'ROC10', 'OBV', 
                            'MFI', 'ROC1', 'ADX', 'Vol.Change', 'TRIX.Sig']]

######Get test data######
data1_test = pd.read_csv('D:\OneDrive\Course\COMP7404\Project\Data\Test_1min.csv',
                   index_col=0, header=0)

Y1_test = np.ravel(data1_test['Ret1Min'])
X1_test = data1_test.loc[:,['ROC5', 'Cl_to_SMA10', 'ATR', 'ROC10', 'OBV', 
                 'MFI', 'ROC1', 'ADX', 'Vol.Change', 'TRIX.Sig']]

######Grid search for suitable parameters######
param_grid = {'n_estimators':[10,30,50,70,100,150],
               'max_features':[3,6,9]}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid = param_grid, return_train_score=True)

start = timeit.default_timer()
grid_search.fit(X1_train, Y1_train)
stop = timeit.default_timer()
print('Time: ', stop - start)

grid_search.cv_results_['params']
grid_search.cv_results_['mean_test_score']

grid_search.best_params_

plt.plot([10,30,50,70,100,150], grid_search.cv_results_['mean_test_score'][0:6],'--bx')

######Prediction######
print('======Prediction for 1-Minute======')
start1 = timeit.default_timer()

clf1 = RandomForestClassifier(n_estimators=150, max_features = 3)
clf1.fit(X1_train, Y1_train)
Y1_pred = clf1.predict(X1_test)

stop1 = timeit.default_timer()
print('Time: ', stop1 - start1) 

print(confusion_matrix(Y1_test, Y1_pred))
print(classification_report(Y1_test, Y1_pred))

print('======Prediction for 5-Minute======')
data5_train = pd.read_csv('D:\OneDrive\Course\COMP7404\Project\Data\Train_5min.csv',
                   index_col=0, header=0)

Y5_train = np.ravel(data5_train['Ret5Min'])       
X5_train = data5_train.loc[:,['ROC5', 'TRIX.Sig', 'ATR', 'ROC10', 'OBV', 
                            'MFI', 'ROC1', 'ADX', 'Cl_to_SMA60', 'Vol.Change']]

#Get test data
data5_test = pd.read_csv('D:\OneDrive\Course\COMP7404\Project\Data\Test_5min.csv',
                   index_col=0, header=0)

Y5_test = np.ravel(data5_test['Ret5Min'])
X5_test = data5_test.loc[:,['ROC5', 'TRIX.Sig', 'ATR', 'ROC10', 'OBV', 
                            'MFI', 'ROC1', 'ADX', 'Cl_to_SMA60', 'Vol.Change']]

start5 = timeit.default_timer()

clf5 = RandomForestClassifier(n_estimators=150, max_features = 3)
clf5.fit(X5_train, Y5_train)
Y5_pred = clf5.predict(X5_test)

stop5 = timeit.default_timer()
print('Time: ', stop5 - start5) 

print(confusion_matrix(Y5_test, Y5_pred))
print(classification_report(Y5_test, Y5_pred))

print('======Prediction for 15-Minute======')
data15_train = pd.read_csv('D:\OneDrive\Course\COMP7404\Project\Data\Train_15min.csv',
                   index_col=0, header=0)

Y15_train = np.ravel(data15_train['Ret15Min'])       
X15_train = data15_train.loc[:,['ROC5', 'Vol.Change', 'ATR', 'ROC10', 'OBV', 
                            'MFI', 'TRIX', 'MACD', 'ADX', 'TRIX.Sig']]

#Get test data
data15_test = pd.read_csv('D:\OneDrive\Course\COMP7404\Project\Data\Test_15min.csv',
                   index_col=0, header=0)

Y15_test = np.ravel(data15_test['Ret15Min'])
X15_test = data15_test.loc[:,['ROC5', 'Vol.Change', 'ATR', 'ROC10', 'OBV', 
                            'MFI', 'TRIX', 'MACD', 'ADX', 'TRIX.Sig']]

start15 = timeit.default_timer()

clf15 = RandomForestClassifier(n_estimators=150, max_features = 3)
clf15.fit(X15_train, Y15_train)
Y15_pred = clf15.predict(X15_test)

stop15 = timeit.default_timer()
print('Time: ', stop15 - start15) 

print(confusion_matrix(Y15_test, Y15_pred))
print(classification_report(Y15_test, Y15_pred))

print('======Prediction for 30-Minute======')
data30_train = pd.read_csv('D:\OneDrive\Course\COMP7404\Project\Data\Train_30min.csv',
                   index_col=0, header=0)

Y30_train = np.ravel(data30_train['Ret30Min'])       
X30_train = data30_train.loc[:,['ROC5', 'MACD', 'ROC10', 'ATR', 'MFI', 
                            'OBV', 'TRIX.Sig', 'TRIX', 'ADX', 'Vol.Change']]

#Get test data
data30_test = pd.read_csv('D:\OneDrive\Course\COMP7404\Project\Data\Test_30min.csv',
                   index_col=0, header=0)

Y30_test = np.ravel(data30_test['Ret30Min'])
X30_test = data30_test.loc[:,['ROC5', 'MACD', 'ROC10', 'ATR', 'MFI', 
                          'OBV', 'TRIX.Sig', 'TRIX', 'ADX', 'Vol.Change']]

start30 = timeit.default_timer()

clf30 = RandomForestClassifier(n_estimators=150, max_features = 3)
clf30.fit(X30_train, Y30_train)
Y_pred = clf30.predict(X30_test)

stop30 = timeit.default_timer()
print('Time: ', stop30 - start30) 

print(confusion_matrix(Y30_test, Y_pred))
print(classification_report(Y30_test, Y_pred))


