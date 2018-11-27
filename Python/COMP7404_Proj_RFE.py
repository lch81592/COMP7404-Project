"""
Objective: This part is to create a algorithm for features selection. We will be using
            recursive features elimination to selectively choose the top 10 features
            using a RF regressor for each target minute. The resultant features are then used 
            running our machine learning. 
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.feature_selection import RFE, f_regression
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, RandomizedLasso)
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

# Import the data
data = pd.read_csv('C:\Users\Roy\Google Drive\COMP7404 Project\Data\Train_1min.csv',
                   index_col=0, header=0)

Y_train = np.ravel(data['Ret1Min'])
Y_new = np.zeros(Y_train.shape[0])
for i in range(Y_train.shape[0]):
    if (Y_train[i] == 'UP'):
        Y_new[i] = 1
    else: 
        if (Y_train[i] == 'DOWN'):
            Y_new[i] = 2

  
X = data.iloc[:,1:27]
y = Y_new

colnames = X.columns

# Define dictionary to store our rankings
ranks = {}
# Create our function which stores the feature rankings to the ranks dictionary
def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))

# Create the random forest regressors and run the RFE with ending features of 10
rf = RandomForestRegressor(n_jobs=-1, n_estimators=25, verbose=3)
rfe = RFE(rf, n_features_to_select=10, verbose =3 )
rfe.fit(X,y)

# Produce the rank of the features
ranks["RFE"] = ranking(list(map(float, rfe.ranking_)), colnames, order=-1)

# Create empty dictionary to store the mean value calculated from all the scores
r = {}
for name in colnames:
    r[name] = round(np.mean([ranks[method][name] 
                             for method in ranks.keys()]), 2)
 
methods = sorted(ranks.keys())
ranks["Mean"] = r
methods.append("Mean")
 
print("\t%s" % "\t".join(methods))
for name in colnames:
    print("%s\t%s" % (name, "\t".join(map(str, 
                         [ranks[method][name] for method in methods]))))
    
# Put the mean scores into a Pandas dataframe
meanplot = pd.DataFrame(list(r.items()), columns= ['Feature','Mean Ranking'])

# Sort the dataframe
meanplot = meanplot.sort_values('Mean Ranking', ascending=False)


# Let's plot the ranking of the features
sns.factorplot(x="Mean Ranking", y="Feature", data = meanplot, kind="bar", 
               size=10, aspect=0.5, palette='coolwarm')
