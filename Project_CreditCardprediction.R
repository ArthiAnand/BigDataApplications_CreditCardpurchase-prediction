# Course : Big Data Applications and Analytics, Indiana University Bloomington
# Code Author :  Arthi Anand 
# Purpose: To predict whether a customer would get a new credit card based on 
# the information collected about him/her.
##############################################################################
getwd()
##Set the file directory accordingly 
setwd("C:\\Users\\ArthiSS\\Desktop\\BDApp")
#Read the csv file as data
data = read.csv("project.csv", header =T, sep = ",")

# Pre-processing the data 
names(data)
str(data)
summary(data)
#Since the median of the instances foreach feature in which NA's are present is 0, 
#Convert the NA's to 0 
data[is.na(data)] <- 0
#Remove attribute "ID" since it's not very informative 
data = data[c(-1)]
#The minimum for experience is -3 which is wrong data entry. The minimum value for experience 
#must be 0. Rename wrong values to 0. 
for(i in 2:nrow(data)) {
  if (data$Experience[i] < 0)
    data$Experience[i] = 0
}
min(data$Experience)
#Check if there is class imbalance
class0 = 0
class1 = 0
for(i in 2:nrow(data)) {
if (data$CreditCard[i] == 0)
  class0 = class0 + 1
else 
  class1 = class1 + 1
}
# There is a class imbalance as suspected 
#  Correlation between the attributes. 
correlation <- cor(data)
install.packages("corrplot")
library(corrplot)
col4 <- colorRampPalette(c("#7F0000", "red", "#FF7F00", "yellow", "#7FFF7F", 
                           "cyan", "#007FFF", "blue", "#00007F"))
corrplot(correlation, method = "shade",type = "lower",order = "hclust", addrect = 2, col = col4(10))

# Since Age and Experience are highly correlated, one of them needs to be removed 
install.packages("mlbench")
install.packages("caret")
install.packages("class")
install.packages("randomForest")
library(mlbench)
library(caret)
library(class)
require(randomForest) # same as library except for its "trying to load package"
fit=randomForest(factor(CreditCard)~., data=data)
varImp(fit)

# Since Age is slightly more important than Experience, remove experience 
data = data[c(-2)]
# Split data into train and test 
rows=seq(1,nrow(data),1)
set.seed(456)
trainRows=sample(rows,(70*nrow(data))/100)
testRows=rows[-(trainRows)]
train = data[trainRows,]
test=data[testRows,]
rm( rows, trainRows, testRows)

# Using Logistic Regression to classify the data
train$CreditCard = as.factor(train$CreditCard)
LogReg <- glm(CreditCard ~., data=train,family=binomial)
install.packages("MASS")
library(MASS)
anova(LogReg, test="Chisq")
step = stepAIC(LogReg,data=train, direction="both")
LogReg1 = glm(CreditCard ~ Mortgage + Personal.Loan + Securities.Account + 
                CD.Account + Online, data=train, family=binomial)

summary(LogReg1)
train_predict<-predict(LogReg1, type="response")

#Classification on train data
train_pred_class <- factor(ifelse(train_predict > 0.50, 1, 0))
#Confusion matrix for prediction on train data

metrics = table(train$CreditCard,train_pred_class)
metrics
# Evaluation metrics calculations based on the confusion matrix. For this data, false negative is more 
#costly. 
accuracy =(metrics[1,1]+metrics[2,2])/(length(train_pred_class))*100
Specificity = metrics[2,2]/(metrics[2,2]+metrics[2,1])*100
#Test data predictions
test$CreditCard = as.factor(test$CreditCard)
test_predict<-predict(LogReg1, newdata= test,type="response")
pred.log <- factor(ifelse(test_predict>0.50, 1, 0))
metrictest = table(test$CreditCard,pred.log)
metrictest
accuracytest =(metrictest[1,1]+metrictest[2,2])/(length(pred.log))*100
Specificitytest = metrictest[2,2]/(metrictest[2,2]+metrictest[2,1])*100
##Check specificity for train and test 
Specificity
Specificitytest
#ROC curve
install.packages("Deducer")
library(Deducer)
rocplot(LogReg1)

#Decision Trees
install.packages("C50")
library(C50)
names(train)
dtC50= C5.0(CreditCard ~ Mortgage + Personal.Loan + Securities.Account + 
              CD.Account + Online,     data = train[],
            rules=TRUE)
summary(dtC50)
C5imp(dtC50, pct=TRUE)
#Print the rules used for classification
dtC50$rules
#Table for train metrics
a=table(train$CreditCard, predict(dtC50, newdata=train, type="class"))
a
#Evaluation metrics for train 
accutraindt =(a[1,1]+a[2,2])/(a[1,1]+a[2,2]+a[1,2]+a[2,1])*100
Specificitytraindt = a[2,2]/(a[2,1]+a[2,2])*100
#Predicting test data
pred.treet = predict(dtC50, newdata=test, type="class")
a1=table(test$CreditCard, predict(dtC50, newdata=test, type="class"))
a1
accutestdt =(a1[1,1]+a1[2,2])/(a1[1,1]+a1[2,2]+a1[1,2]+a1[2,1])*100
Specificitytestdt = a1[2,2]/(a1[2,2]+a1[2,1])*100
Specificitytestdt
rm(a,a1)

###########################
#SVM
table(data$CreditCard)
install.packages("e1071")
library(e1071)
model <- svm(CreditCard ~ ., data = train)
print(model)
summary(model)

# predict with train data
pred_svm <- predict(model, train)
trainclass = (train$CreditCard)
pred_train = table(pred_svm, trainclass)

# Metrics for train:
accuracysvmtrain=(pred_train[2,2]+pred_train[1,1])/(pred_train[2,1]+pred_train[2,2]+pred_train[1,1]+pred_train[1,2])*100
Specificitysvmtrain = pred_train[2,2]/(pred_train[2,1]+ pred_train[2,2])*100
# predict with test:
pred_svm_test <- predict(model, test)
testclass = (test$CreditCard)
pred_test = table(pred_svm_test, testclass)
# Metrics for test:
accuracysvmtest=(pred_test[2,2]+pred_test[1,1])/(pred_test[2,1]+pred_test[2,2]+pred_test[1,1]+pred_test[1,2])*100
Specificitysvmtest = pred_test[2,2]/(pred_test[2,1]+ pred_test[2,2])*100

#### Naive Bayes
library(e1071)
model <- naiveBayes(CreditCard ~ ., data = train)
pred_train <- predict(model, train)
pred_test <- predict(model, test)
nbtrain <- table(pred_train, train$CreditCard)
#Metrics for train 
accuracynbtrain=(nbtrain[2,2]+nbtrain[1,1])/(nbtrain[2,1]+nbtrain[2,2]+nbtrain[1,1]+nbtrain[1,2])*100
Specificitynbtrain = nbtrain[2,2]/(nbtrain[2,2]+nbtrain[2,1])*100
nbtest <- table(pred_test, test$CreditCard)
#Metrics for test
accuracynbtest = (nbtest[2,2]+nbtest[1,1])/(nbtest[2,1]+nbtest[2,2]+nbtest[1,1]+nbtest[1,2])*100
Specificitynbtest = nbtest[2,2]/(nbtest[2,2]+nbtest[2,1])*100
## using laplace smoothing:
model <- naiveBayes(CreditCard ~ ., data = train, laplace = 10)
pred <- predict(model, test[,-12])
nb_Test_smoothing <- table(pred, test$CreditCard)
#Metrics for smoothing:
accuracynbtest = (nb_Test_smoothing[2,2]+nb_Test_smoothing[1,1])/(nb_Test_smoothing[2,1]+nb_Test_smoothing[2,2]+nb_Test_smoothing[1,1]+nb_Test_smoothing[1,2])*100
Specificitynbtest = nb_Test_smoothing[2,2]/(nb_Test_smoothing[2,2]+nb_Test_smoothing[2,1])*100
              
#Visualization: informative & important facets
data$CreditCard = as.factor(data$CreditCard)
data$CD.Account = factor(data$CD.Account)

install.packages("ggplot2")
library(ggplot2)
# Credit card purchase based on family size, personal loan, cd account
ggplot(train, aes(CD.Account, fill=CreditCard)) + geom_bar() + 
  ggtitle("Credit card purchase based on family size, personal loan, CD.Account")+scale_fill_hue(c=45, l=80)+ scale_fill_manual(values=c( "#660033","#FF0000")) +
  facet_grid(Personal.Loan ~ Family)
# Securities. Account and Online intereaction and Credit Card buyers prediction
p = qplot(Online,Securities.Account, data=train,
          color=CreditCard,  size=I(3))
q = p + ggtitle("Securities. Account and Online intereaction and Credit Card buyers prediction") 
print(q)
#Credit card buyers based Age and Education
ggplot(train, aes(Age, fill=CreditCard)) + geom_bar() + 
  ggtitle("Credit card buyers based Age and Education") +
  scale_fill_manual(values=c( "#660033","#000033")) +
  facet_grid(Personal.Loan ~ Education)
#Credit Card buyers based on Income 
m <- ggplot(train, aes(Income, fill=CreditCard))
m + geom_bar()+ ggtitle("Credit Card buyers based on Income") + 
  scale_fill_manual(values=c( "#660033","#CC0066"))
###NOTE: the message in red "stat_bin: 
### binwidth defaulted to range/30. Use 'binwidth = x' to adjust this" 
##  is not an error or a warning. It's just a message. It can be suppressed. 
## I chose not to supress the message. 
