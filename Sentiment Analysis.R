#Import the dataset
data = read.delim('Restaurant_Reviews.tsv', quote = '', stringsAsFactors = FALSE)
colnames(data)[2] = c('Sentiment')
                  
#Basic information about the dataset
dim(data)
str(data)
table(data$Sentiment)
typeof(data)
class(data)

#Import the require libraries
library(tm)
library(SnowballC)
corpus = VCorpus(VectorSource(data$Review))
corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeWords, stopwords())

corpus = tm_map(corpus, stemDocument)
corpus = tm_map(corpus, stripWhitespace)

#creating the Bag of words model
dtm = DocumentTermMatrix(corpus)
dtm = removeSparseTerms(dtm, 0.999) #remove the words with less % then 99.9%

dtm_mat = as.matrix(dtm)

#sort the matrix
sorted_dtm = dtm_mat[, order(colSums(dtm_mat), decreasing = TRUE)]

#lets try to visualize the relation
sum = colSums(sorted_dtm)
barplot(sum, 
        main = "Count Words vs Words",
        xlab ="Words",
        ylab = "Count",
        col = 'skyblue',
        las = 2,
        ylim = c(0, max(sum) +1))


#create the Dataframe of the matrix
dataset = as.data.frame(dtm_mat)

dim(dataset)
str(dataset)
table(data)
typeof(dataset)
class(dataset)

#Append the Dependent column the new data frame
dataset$Sentiment = data$Sentiment

#Encoding the category feature as factor
dataset$Sentiment = factor(dataset$Sentiment, levels = c(0, 1))

#split the dataset into training and testing
library(caTools)
set.seed(123)
split = sample.split(dataset$Sentiment, SplitRatio = 0.8)
train = subset(dataset, split == TRUE)
test = subset(dataset, split == FALSE)

#Choose a algorithm for the classify the category
#picking the random Forest Method
#import the require library
library(randomForest)
classifier = randomForest(x = train[-ncol(train)],
                          y = train$Sentiment, ntree = 20)

#Predict the test set result
y_pred = predict(classifier, newdata = test[-ncol(test)])

#Visualize the outcome
plot(classifier)

#Results: make the confusion matrix for Random Forest
cm = table(test$Sentiment, y_pred)
accuracy = paste("Accuracy: ", 100*sum(diag(cm))/sum(cm),"%")
precision = paste("Precision: ", cm[2,2]/sum(cm[1,]))
recall = paste("Recall: ",cm[2,2]/sum(cm[,1]))
F1score = paste("F1Score: ", 2*(cm[2,2]/sum(cm[1,]) * cm[2,2]/sum(cm[,1]))/(cm[2,2]/sum(cm[1,]) + cm[2,2]/sum(cm[,1])))

#lets try to apply Logistic regression
classiLog = glm(formula = Sentiment ~., data = train, family = binomial)
prob_pred = predict(classiLog, newdata = test[-ncol(test)], type = 'response')
y_pred = ifelse(prob_pred>0.5, 1, 0)
cm=  table(test[,ncol(test)], y_pred)

#Results: make the confusion matrix for Loistic Regression
cm = table(test$Sentiment, y_pred)
accuracy = paste("Accuracy: ", 100*sum(diag(cm))/sum(cm),"%")
precision = paste("Precision: ", cm[2,2]/sum(cm[1,]))
recall = paste("Recall: ",cm[2,2]/sum(cm[,1]))
F1score = paste("F1Score: ", 2*(cm[2,2]/sum(cm[1,]) * cm[2,2]/sum(cm[,1]))/(cm[2,2]/sum(cm[1,]) + cm[2,2]/sum(cm[,1])))

#let's now to apply Naive Bayes
library(e1071)
classiNai = naiveBayes(x = train[-ncol(train)],
                       y = train$Sentiment)
y_pred = predict(classiNai, newdata = test[-ncol(test)])
cm = table(test[,ncol(test)], y_pred)

#Results: make the confusion matrix for Naive Bayes
cm = table(test$Sentiment, y_pred)
accuracy = paste("Accuracy: ", 100*sum(diag(cm))/sum(cm),"%")
precision = paste("Precision: ", cm[2,2]/sum(cm[1,]))
recall = paste("Recall: ",cm[2,2]/sum(cm[,1]))
F1score = paste("F1Score: ", 2*(cm[2,2]/sum(cm[1,]) * cm[2,2]/sum(cm[,1]))/(cm[2,2]/sum(cm[1,]) + cm[2,2]/sum(cm[,1])))

#predicting the sentiment of the given manual input review
classify_sentiment <- function(input_review, classifier, dtm_vocab) {
  # Create a Corpus containing only the input review
  input_corpus <- Corpus(VectorSource(input_review))
  
  # Preprocess the input review
  input_corpus <- tm_map(input_corpus, content_transformer(tolower))
  input_corpus <- tm_map(input_corpus, removeNumbers)
  input_corpus <- tm_map(input_corpus, removePunctuation)
  input_corpus <- tm_map(input_corpus, removeWords, stopwords())
  input_corpus <- tm_map(input_corpus, stemDocument)
  input_corpus <- tm_map(input_corpus, stripWhitespace)
  
  
  # Create a document-term matrix for the input review
  input_dtm <- DocumentTermMatrix(input_corpus, control = list(dictionary = dtm_vocab))
  input_dtm <- as.matrix(input_dtm)
  
  # Predict sentiment using the trained classifier
  y_pred <- predict(classifier, newdata = input_dtm)
  
  # Return the predicted sentiment
  return(y_pred)
}

input_review <- "Today"
predicted_sentiment <- classify_sentiment(input_review, classifier, dtm_vocab = colnames(dtm))
ifelse(predicted_sentiment==1, "Positive", "Negative")


# # Example confusion matrix (replace this with your actual confusion matrix)
# confusion_matrix <- matrix(c(79, 21, 27, 73), nrow = 2, byrow = TRUE)
# 
# # Create a heatmap of the confusion matrix
# heatmap(confusion_matrix, 
#         Rowv = NA, Colv = NA,
#         col = cm.colors(256),
#         scale = "column",
#         margins = c(5, 10),
#         xlab = "Predicted Sentiment",
#         ylab = "True Sentiment",
#         main = "Confusion Matrix Heatmap")

