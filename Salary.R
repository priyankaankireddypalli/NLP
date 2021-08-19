# 1
install.packages("naivebayes")
library(naivebayes)
library(ggplot2)
install.packages("caret")
library(caret)
install.packages("psych")
library(psych)
install.packages("e1071")
library(e1071)
# Importing Training dataset
train <- read.csv("C:\\Users\\WIN10\\Desktop\\LEARNING\\NaiveBayesProblemStatement\\SalaryData_Train.csv")
str(train)
View(train)
train$educationno <- as.factor(train$educationno)
class(train)
# Importing Test dataset
test <- read.csv("C:\\Users\\WIN10\\Desktop\\LEARNING\\NaiveBayesProblemStatement\\SalaryData_Test.csv")
str(test)
View(test)
test$educationno <- as.factor(test$educationno)
class(test)
# Naive Bayes Model 
Model <- naiveBayes(as.factor(train$Salary) ~ ., data = train)
Model
# Classification Model
Model_pred <- predict(Model,test)
mean(Model_pred==test$Salary)
confusionMatrix(Model_pred,as.factor(test$Salary))
