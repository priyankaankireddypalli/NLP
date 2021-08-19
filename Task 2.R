# Task 2
library(rvest)
library(XML)
library(magrittr)
# Importing IMDB reviews
iurl <- "https://www.imdb.com/title/tt1798709/reviews?ref_=tt_ql_3"
imdb_reviews <- NULL
for (i in 1:10){
  murl <- read_html(as.character(paste(iurl,i,sep="=")))
  rev <- murl %>% html_nodes(".show-more__control") %>% html_text()
  imdb_reviews <- c(imdb_reviews,rev)
}
length(imdb_reviews)
write.table(imdb_reviews,"her.txt",row.names = F)
getwd()
her <- read.delim('her.txt')
str(her)
View(her)
# Performing sentiment analysis
txt <- imdb_reviews
library(tm)
str(txt)
# Converting character data to corpus
x <- Corpus(VectorSource(txt))
inspect(x[1:5])
x <- tm_map(x, function(x) iconv(enc2utf8(x), sub='byte'))
# Performing data cleansing
x <- tm_map(x,tolower)    # Converting to lower case
inspect(x[1])
x <- tm_map(x,removePunctuation)  # Removing punctuations
inspect(x[1])
x <- tm_map(x,removeWords,stopwords('english'))  # Removing stopwords
inspect(x[1])
x <- tm_map(x,removeNumbers)  # Removing numbers
inspect(x[1])
x <- tm_map(x,stripWhitespace)  # Stripping out the spaces
inspect(x[1])
removeURL <- function(x) gsub('http[[:alnum:]]*','',x)
x <- tm_map(x, content_transformer(removeURL))
inspect(x[1:5])

# Term Document Matrix
# Converting unstructured data to structured
tdm <- TermDocumentMatrix(x)
tdm
dtm <- t(tdm)
dtm <- DocumentTermMatrix(x)
# Removing sparse entries
corpus_dtm_freq <- removeSparseTerms(tdm,0.99)
tdm <- as.matrix(tdm)
dim(tdm)
tdm[1:20,1:20]
# Bar plot
w <- rowSums(tdm)
w
w_sub <- subset(w,w>=65)
w_sub
barplot(w_sub,las=2,col = rainbow(30))
x <- tm_map(x, removeWords,c('just','will'))   # 'just' 'will' are common words
x <- tm_map(x, stripWhitespace)
tdm <- TermDocumentMatrix(x)
tdm <- as.matrix(tdm)
tdm[100:109,1:20]
w <- rowSums(tdm)
w_sub <- subset(w,w>=50)
barplot(w_sub,las = 2,col = rainbow(30))
# Word cloud
library(wordcloud)
wordcloud(words = names(w_sub),freq = w_sub)
w_sub1 <- sort(rowSums(tdm),decreasing = TRUE)
head(w_sub1)
wordcloud(words = names(w_sub1),freq = w_sub1)
# For better visualization
wordcloud(words = names(w_sub1),freq = w_sub1,random.order = FALSE,colors = rainbow(30),scale = c(2,0.5),rot.per = 0.4)
# Word cloud 2
library(wordcloud2)
w1 <- data.frame(names(w_sub),w_sub)
colnames(w1) <- c('word','freq')
wordcloud2(w1,size = 0.3,shape = 'triangle')
# Bigram


library(RWeka)
library(wordcloud)
minfreqbigram <- 2
bitoken <- NGramTokenizer(x,Weka_control(min = 2,max = 2))
two_words <- data.frame(table(bitoken))
sort_two <- two_words[order(two_words$Freq,decreasing = TRUE), ]
wordcloud(sort_two$bitoken,sort_two$Freq,random.order = F,scale=c(2,0.35),min.freq = minfreqbigram,colors = brewer.pal(8,'Dark2'),max.words = 150)
# Sentiment Analysis for tweets:
install.packages("syuzhet")
library(syuzhet)
install.packages("lubridate")
library(lubridate)
library(ggplot2)
library(scales)
library(reshape2)
library(dplyr)
# Read File 

imdb_reviews <- read.delim('her.TXT')
reviews <- as.character(imdb_reviews[-1,])
class(reviews)
# Obtain Sentiment scores 

s <- get_nrc_sentiment(reviews)
head(s)
reviews[4]
get_nrc_sentiment('splendid')
get_nrc_sentiment('no words') 
# barplot 

barplot(colSums(s), las = 2.5, col = rainbow(10),ylab = 'Count',main= 'Sentiment scores for IMDB Reviews for Her')
