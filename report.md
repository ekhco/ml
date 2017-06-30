# The Quantified Self Movement: Predicting Poor Exercise Form
Eric Chow  
Jun 29, 2017  

## Synopsis
People regularly quantify how much of an activity they perform, but they do not quantify how well they perform them. In this analysis, I look at data from belt, forearm, arm, and dumbell accelerometers to predict whether they perform the activity correctly or not in 5 different ways. The data is available from (here) [http://groupware.les.inf.puc-rio.br/har].  I first drop any columns with missing or #DIV/0!. I split the data into a training and test set (75/25) and pre-process by centering and scaling each variable. Random forest was initially going to be used as the prediction model, but it took so long, so I switched it to GBM (generalized boosted models). I considered ensemble methods, but after running a random forest and seeing how long it took, I decided against it due to time constraints and feasibility. When the model was used against the validation dataset, the accuracy was 35% (7/20) compared to the in-sample rate of 100%! So the in-sample rate was very not reflective of out-of-sample error. I clearly overfit the data.  Given the time constraints, it is very apparent that machine learning and predictive modelling can take a lot of work.


```r
rm(list=ls())
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
setwd("~/Dropbox/coursera/machine_learning/ml")
quiz <- read.csv("pml-testing.csv")
data <- read.csv("pml-training.csv")
```

## Data PreProcessing
I drop any columns with NA or NUM/0! manually. I also use centering and scaling for preprocessing as the medians and variances are quite heterogeneous across the measures. I use the caret package to do this.


```r
# drop funny columns (NA, or #DIV/0!)
data_ <- data[,!sapply(data,function(x) any(is.na(x)))]
data__ <- data_[,!sapply(data_,function(x) any(x=="#DIV/0!"))]
```

## Subsetting Data
First, I split the training file into a training (75%) and testing (25%) set because the test file is actually for the quiz (validation).

```r
in_train <- createDataPartition(y=data__$classe, p=0.75, list=FALSE)
trn <- data__[in_train, ]
tst <- data__[-in_train, ]
```

## Generalized Boosting Models Prediction
A random forest approach was attempted to predict the "classe."  It took too long to run. So I killed it, and switched the method to GBM. That also took a long time. I let it run overnight. I saved the model to a file so I can load it quickly.


```r
# modelFit <- train(classe ~ ., preProcess=c("center","scale"), data=trn, method = "gbm")
# saveRDS(modelFit, "gbm_fit.rds")
modelFit <- readRDS("gbm_fit.rds") # I ran the model previously, and now I load it
```

## Prediction, Testing, Validation
When we test the prediction against the 25% test set, we find that the accuracy is way too high. I have overfit this data.  I don't expect my out of sample error rate to be acceptable. I subsetted the columns of the quiz dataset to the same columns I used and applied the prediction model. In fact, It turns out with after validation, that the prediction is a all class "A."

```r
prd <- predict(modelFit, newdata=tst)
```

```
## Loading required package: gbm
```

```
## Loading required package: survival
```

```
## 
## Attaching package: 'survival'
```

```
## The following object is masked from 'package:caret':
## 
##     cluster
```

```
## Loading required package: splines
```

```
## Loading required package: parallel
```

```
## Loaded gbm 2.1.3
```

```
## Loading required package: plyr
```

```r
# use confusion matrix to EVALUATE
confusionMatrix(prd, tst$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    0    0    0    0
##          B    0  949    0    0    0
##          C    0    0  855    0    0
##          D    0    0    0  804    1
##          E    0    0    0    0  900
## 
## Overall Statistics
##                                      
##                Accuracy : 0.9998     
##                  95% CI : (0.9989, 1)
##     No Information Rate : 0.2845     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 0.9997     
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   0.9989
## Specificity            1.0000   1.0000   1.0000   0.9998   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   0.9988   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   0.9998
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2845   0.1935   0.1743   0.1639   0.1835
## Detection Prevalence   0.2845   0.1935   0.1743   0.1642   0.1835
## Balanced Accuracy      1.0000   1.0000   1.0000   0.9999   0.9994
```

```r
quiz_ <- quiz[,names(data__)[-60]]
prd <- predict(modelFit, newdata=quiz_)
prd
```

```
##  [1] A A A A A A A A A A A A A A A A A A A A
## Levels: A B C D E
```

Haha! Why did I need to run such a model for so long, only for it to predict all "A"? Clearly, this is only a beginning. I need to revisit the model and check for overfitting.  There are a variety of variables that I should exclude, like "X" which is just a number sequence, user_name perhaps, raw_timestamp, and new_window.  Predicting on roll, pitch, yaw, acceleration, gyros alone might provide better out-of-sample error.

## Conclusions

From our prediction analysis, we can see that it is very easy to overfit data that produces predictions that have no generalizability. Time constraints are also a significant factor. If I had more time, I would have considered the residuals better and looked at the variable importance to understand which variables to include or not.
