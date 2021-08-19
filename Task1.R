# Task 1
#### Amazon ####
library(rvest)
library(XML)
library(magrittr)
# importing reviews from amazon
aurl <- "https://www.amazon.in/New-Apple-iPhone-Mini-128GB/product-reviews/B08L5VN68Y/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews"
amazon_reviews <- NULL
for (i in 1:20) {
  murl <- read_html(as.character(paste(aurl,i,sep="=")))
  rev <- murl %>% html_nodes(".review-text") %>% html_text()
  amazon_reviews <- c(amazon_reviews,rev)
}
write.table(amazon_reviews,"Iphone 12 mini.txt")
getwd()
# Performing Sentiment Analysis
txt <- amazon_reviews
str(txt)
length(txt)
View(txt)
library(tm)
# Converting character data to corpus data
x <- Corpus(VectorSource(txt))
inspect(x[1])
x <- tm_map(x, function(x) iconv(enc2utf8(x), sub='byte'))
# Performing data cleansing
x1 <- tm_map(x,tolower)  # Converting to lower case
inspect(x1[1])
x1 <- tm_map(x1,removePunctuation)  # Removing all the punctuations
inspect(x1[1])
x1 <- tm_map(x1,removeWords,stopwords('english'))   # Removing stop words
inspect(x1[1])
x1 <- tm_map(x1,removeNumbers)  # Removing the numbers
inspect(x1[1])
x1 <- tm_map(x1,stripWhitespace)  # Stripping out the spaces
inspect(x1[1])
# Term Document Matrix
# Converting unstructured data to structured
tdm <- TermDocumentMatrix(x1)
dtm <- t(tdm)
dtm <- DocumentTermMatrix(x1)

# Removing sparse entries 
corpus_dtm_frequent <- removeSparseTerms(tdm,0.99)
tdm <- as.matrix(tdm)
dim(tdm)
tdm[1:20,1:200]
inspect(x[1])
# Bar plot
w <- rowSums(tdm)
w
w_sub <- subset(w,w>=65)
w_sub
windows()
barplot(w_sub,las=2,col = rainbow(30))
x1 <- tm_map(x1, stripWhitespace)
tdm <- TermDocumentMatrix(x1)
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
install.packages("Rweka")
library(RWeka)
library(wordcloud)
minfreqbigram <- 2
bitoken <- NGramTokenizer(x1,Weka_control(min = 2,max = 2))
two_words <- data.frame(table(bitoken))
sort_two <- two_words[order(two_words$Freq,decreasing = TRUE), ]
wordcloud(sort_two$bitoken,sort_two$Freq,random.order = F,scale=c(2,0.35),min.freq = minfreqbigram,colors = brewer.pal(8,'Dark2'),max.words = 150)
# Loading Positive and Negative words
positive <- readLines(file.choose())
negative <- readLines(file.choose())
stopwords <- readLines(file.choose())
# Positive word cloud
positivematch <- match(names(w_sub1),positive)
positivematch <- !is.na(positivematch)
freqpos <- w_sub1[positivematch]
names <- names(freqpos)
wordcloud(names,freqpos,scale = c(4,1),colors = brewer.pal(8,'Dark2'))
# Negative word cloud
negativematch <- match(names(w_sub1),negative)
negativematch <- !is.na(negativematch)
freqneg <- w_sub1[negativematch]
names <- names(freqneg)
wordcloud(names,freqneg,scale = c(4,.5),colors = brewer.pal(8,'Dark2'))
