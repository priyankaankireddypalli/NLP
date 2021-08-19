# 2
import numpy as np
from sklearn.model_selection import train_test_split as tts
import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
suv = pd.read_csv('C:\\Users\\WIN10\\Desktop\\LEARNING\\NaiveBayesProblemStatement\\NB_Car_Ad.csv')
# X is the matrix of features, it contains independent variable number 2 and 3 which is Age,EstimatedSalary according to dataset
X = suv.iloc[:,[2,3]].values
# Y contains dependent variable which is Purchased according to dataset and the column number is 4
Y = suv.iloc[:,4].values
# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = tts(X,Y,test_size=0.2, random_state=0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Fitting classifier to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train , y_train)
# Predicting the Test set results
prediction = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, prediction)
confusion_matrix
# Calculating the accuracy of this model w.r.t. this dataset
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, prediction))
