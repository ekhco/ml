rm(list=ls())
library(caret)
setwd("~/Dropbox/coursera/machine_learning/ml")
quiz <- read.csv("pml-testing.csv")
data <- read.csv("pml-training.csv")




# drop funny columns (NA, or #DIV/0!)
data_ <- data[,!sapply(data,function(x) any(is.na(x)))]
data__ <- data_[,!sapply(data_,function(x) any(x=="#DIV/0!"))]



# subset using partition
in_train <- createDataPartition(y=data__$classe, p=0.75, list=FALSE)
trn <- data__[in_train, ]
tst <- data__[-in_train, ]




# preprocess and train
modelFit <- train(classe ~ . , data=trn, preProcess=c("center","scale"), method = "rf")




# prediction
prd <- predict(modelFit, newdata=tst)
prd
# use confusion matrix to EVALUATE
confusionMatrix(prd, tst$classe)
