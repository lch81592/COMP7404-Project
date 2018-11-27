import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

def load_data(interval, selected_features):
	train_data = pd.read_csv("Train_" + interval.lower() + ".csv", header=0)
	train_X = train_data[selected_features]
	train_X = np.array(train_X)
	train_y_orig = np.ravel(train_data["Ret" + interval])
	train_y = np.zeros(train_y_orig.shape[0])
	for i in range(train_y_orig.shape[0]):
		if train_y_orig[i] == "UP":
			train_y[i] = 1
		elif train_y_orig[i] == "DOWN":
			train_y[i] = -1
		else:
			train_y[i] = 0

	test_data = pd.read_csv("Test_" + interval.lower() + ".csv", header=0)
	test_X = test_data[selected_features]
	test_X = np.array(test_X)
	test_y_orig = np.ravel(test_data["Ret" + interval])
	test_y = np.zeros(test_y_orig.shape[0])

	up_cnt = 0; down_cnt = 0; flat_cnt = 0
	for i in range(test_y_orig.shape[0]):
		if test_y_orig[i] == "UP":
			test_y[i] = 1
			up_cnt+=1
		elif test_y_orig[i] == "DOWN":
			test_y[i] = -1
			down_cnt+=1
		else:
			test_y[i] = 0
			flat_cnt+=1

	# print("up %d : down %d : flat %d" % (up_cnt, down_cnt, flat_cnt))
	return train_X, train_y, test_X, test_y

def calc(interval, selected_features):
	print("interval %s" % interval)

	train_X, train_y, test_X, test_y = load_data(interval, selected_features)

	# Initialize the MLPClassifier with the parameters
	clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(50, ))

	# Create a model by fitting the training data to the classifier (≈ 1 line)
	# y needs to be flattened before training by using the ravel() function
	clf.fit(train_X, np.ravel(train_y))

	# Predict on new testing data (≈ 1 line)
	pred_y = clf.predict(test_X)

	print("Accuracy on testing data = " + str(clf.score(test_X, test_y)))
	print("Confusion Matrix:")
	print(confusion_matrix(test_y, pred_y))
	# print(classification_report(test_y, pred_y))

	print("=======================================================")

if __name__ == '__main__':
	print("+++ Neural Network +++")
	calc("1Min",  ["ROC5", "Cl_to_SMA10", "ROC10", "ATR", "MFI", "OBV", "ROC1", "ADX", "Vol.Change"])
	calc("5Min",  ["ROC5", "TRIX.Sig", "ATR", "ROC10", "OBV", "MFI", "ROC1", "ADX", "Cl_to_SMA60", "Vol.Change"])
	calc("15Min", ["ROC5", "Vol.Change", "ATR", "ROC10", "OBV", "MFI", "TRIX", "MACD", "ADX", "TRIX.Sig"])
	calc("30Min", ["ROC5", "MACD", "ROC10", "ATR", "MFI", "OBV", "TRIX.Sig", "TRIX", "Vol.Change", "ADX"])
