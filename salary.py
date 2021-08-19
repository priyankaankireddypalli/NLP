# 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Importing train dataset
train = pd.read_csv('C:\\Users\\WIN10\\Desktop\\LEARNING\\NaiveBayesProblemStatement\\SalaryData_Train.csv')
# Importing test dataset
test = pd.read_csv('C:\\Users\\WIN10\\Desktop\\LEARNING\\NaiveBayesProblemStatement\\SalaryData_Test.csv')
train.columns
test.columns
string_columns = ['workclass','education','maritalstatus','occupation','relationship','race','sex','native']
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
for i in string_columns:
    train[i] = label_encoder.fit_transform(train[i])
    test[i] = label_encoder.fit_transform(test[i])
col_names = list(train.columns)
train_X = train[col_names[0:13]]
train_Y = train[col_names[13]]
test_x = test[col_names[0:13]]
test_y = test[col_names[13]]
# Naive Bayes 
# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
Gmodel = GaussianNB()
train_pred_gau = Gmodel.fit(train_X,train_Y).predict(train_X)
test_pred_gau = Gmodel.fit(train_X,train_Y).predict(test_x)
train_acc_gau = np.mean(train_pred_gau==train_Y)
test_acc_gau = np.mean(test_pred_gau==test_y)
train_acc_gau
test_acc_gau
# Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB
Mmodel = MultinomialNB()
train_pred_multi = Mmodel.fit(train_X,train_Y).predict(train_X)
test_pred_multi = Mmodel.fit(train_X,train_Y).predict(test_x)
train_acc_multi = np.mean(train_pred_multi==train_Y)
test_acc_multi = np.mean(test_pred_multi==test_y)
train_acc_multi
test_acc_multi
