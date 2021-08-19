# 2
library(readr)
suv <- read.csv('C:\\Users\\WIN10\\Desktop\\LEARNING\\NaiveBayesProblemStatement\\NB_Car_Ad.csv')
View(suv)
str(suv)
suv$Purchased <- factor(suv$Purchased)
suv$EstimatedSalary <- as.character(suv$EstimatedSalary)
# Examine the type variable more carefully
str(suv$Purchased)
table(suv$Purchased)
# Proportion of car being purchased or not.
prop.table(table(suv$Purchased))
# Build a corpus using the text mining (tm) package
library(tm)
str(suv$EstimatedSalary)
suvcorpus <- Corpus(VectorSource(suv$EstimatedSalary))
suvcorpus <- tm_map(suvcorpus, function(x) iconv(enc2utf8(x), sub='byte'))
# Data Cleansing
# To remove the white spaces in our text  
corpusclean <- tm_map(suvcorpus, stripWhitespace)
inspect(corpus_clean[1])
# Create a document-term sparse matrix
suvdtm <- DocumentTermMatrix(corpusclean)
suvdtm
View(suvdtm[1:10, 1:30])
# To view DTM we need to convert it into matrix first
dtmmatrix <- as.matrix(suvdtm)
str(dtmmatrix)
View(dtmmatrix[1:10, 1:20])
colnames(suvdtm)[1:50]
# Creating training and test datasets
suvtrain <- suv[1:385, ]
suvtest  <- suv[25:400, ]
suvcorpustrain <- suvcorpus[1:385]
suvcorpustest  <- suvcorpus[25:400]
suvdtmtrain <- suvdtm[1:385, ]
suvdtmtest  <- suvdtm[25:400, ]
# Check that the proportion of car Ad we have splited with respect to the population.
prop.table(table(suv$Purchased))
prop.table(table(suvtrain$Purchased))
prop.table(table(suvtest$Purchased))
# Indicator features for frequent words
# dictionary of words which are used more than 2 times
suvdict <- findFreqTerms(suvdtmtrain , 2)
cars_train <- DocumentTermMatrix(suvcorpustrain, list(dictionary = suvdict))
cars_test  <- DocumentTermMatrix(suvcorpustest, list(dictionary = suvdict))
cars_test_matrix <- as.matrix(cars_test)
View(cars_test_matrix [1:10,1:10])
# convert counts to a factor
# custom function: if a word is used more than 0 times then mention 1 else mention 0
convertcounts <- function(x) {
  x <- ifelse(x > 0, 1, 0)
  x <- factor(x, levels = c(0, 1), labels = c("No", "Yes"))
}
# apply() convert_counts() to columns of train/test data
cars_train <- apply(cars_train, MARGIN = 2, convertcounts)
cars_test  <- apply(cars_test, MARGIN = 2, convertcounts)
View(cars_test[1:10,1:10])
# Training a model on the data
library(e1071)
# building naiveBayes classifier.
cars_classifier <- naiveBayes(cars_train, suvtrain$Purchased)
cars_classifier
# Evaluating model performance on train data
carstrainpred <- predict(cars_classifier, cars_train)
# train accuracy
train_acc = mean(carstrainpred == suvtrain$Purchased)
train_acc
# Evaluating model performance on test data
carstestpred <- predict(cars_classifier, cars_test)
# test accuracy
test_acc <- mean(carstestpred == suvtest$Purchased)
test_acc
# crosstable for test model
installed.packages("gmodels")
library(gmodels)
CrossTable(carstestpred, suvtest$Purchased,prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,dnn = c('predicted', 'actual'))
# laplace smoothing, by default the laplace value = 0
# naiveBayes function has laplace parameter, the bigger the laplace smoothing value, 
# the models become same.
carslap <- naiveBayes(cars_train, suvtrain$Purchased,laplace = 2)
carslap
# Evaluating model performance after applying laplace smoothing
carstestpredlap <- predict(carslap, cars_test)
# crosstable of laplace smoothing model
CrossTable(carstestpredlap, suvtest$Purchased,prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,dnn = c('predicted', 'actual'))
# test accuracy after laplace 
testacclap <- mean(carstestpredlap == suvtest$Purchased)
testacclap
#we are getting the train accuracy as 79% and test accuracy as 77.6 so we are 
#getting the difference as 2% which is accepted.
