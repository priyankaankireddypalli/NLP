# 3
install.packages("klaR")
require("klaR")
library(readr)
library(purrr)
library(dplyr)
library(caTools)
install.packages("extraTrees")
library(extraTrees)
library(MASS)
install.packages("randomForest")
library(randomForest)
library(caret)
install.packages("ranger")
library(ranger)
library(ModelMetrics)
install.packages("xgboost")
library(xgboost)
library(e1071)
library(tm)
# reading the file
tweets <- read_csv('C:\\Users\\WIN10\\Desktop\\LEARNING\\NaiveBayesProblemStatement\\Disaster_tweets_NB.csv')
head(tweets)
# Checking for NA values
sum(is.na(tweets))
# NA values are present in our dataset
# Replacing the NA values
map_dbl(tweets, ~ sum(is.na(.))/nrow(tweets)*100)
tweets <- tweets %>% dplyr::select(-c(id, keyword, location))
head(tweets)
tweets %>% group_by(target) %>% tally() %>% mutate(Percentage=n/sum(n)*100) %>% round(digits = 2)
sum(is.na(tweets))
# We have successfully removed NA values
# Data cleansing
corpus <- tm::VCorpus(tm::VectorSource(tweets$text))
corpus.ws_rm <- tm::tm_map(corpus, tm::stripWhitespace)  # Removing whitespaces
corpus.tolower <- tm::tm_map(corpus.ws_rm , tm::content_transformer(tolower))   # Converting to lowercase
control_list <- list(removePunctuation = TRUE,stopwords = TRUE,stemming = TRUE)  # Removing punctuations
update.packages("tm",  checkBuilt = TRUE)
install.packages("SnowballC")
dtm <- tm::DocumentTermMatrix(corpus.tolower, control = control_list)
dtm
tm::inspect(dtm[1:13, 1:15])
# remove sparse words that appears less than 1% of the time. Essentially setting the sparsity thresholds at 99%
dtm <- tm::removeSparseTerms(dtm, sparse = 0.999) 
# Converting DocumentTermMatrix into a data frame 
tweet_final_df <- data.frame(as.matrix(dtm), stringsAsFactors = FALSE)
# Adding the target column to enable modeling. Each obs in tweet_final_df is at the tweet level.
tweet_final_df$target <- tweets$target
set.seed(8898)
split = sample.split(tweet_final_df, SplitRatio = 0.80)
train_df = subset(tweet_final_df, split == TRUE)
# Naïve Bayes Classifier
# Create response and feature data
features <- setdiff(names(train_df), "Target")
x <- train_df[, features]
y <- train_df$target
time <- system.time({
  NBC <- naiveBayes(train_df[, features], as.factor(y))
})
cat("Time taken is:", sum(time[2] + time[3]), " seconds.")
NBC_pred <-predict(NBC,train_df[, features])
caret::confusionMatrix(data = NBC_pred, reference = as.factor(train_df$target), positive = "1")

