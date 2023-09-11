#Load libraries

library(psych)
library(ppcor)
library(nFactors)
library(GPArotation)
library(cluster)
library(fpc)
library(car)
library(relaimpo)
library(RcmdrMisc)
library(pROC)
library(ROCR)

#select the directory

setwd("C:\\Users\\Gifty Aiyegbeni\\Desktop\\S 2\\ADM\\Submission")

getwd()

#read the data

data = read.csv("SouthGermanCredit.csv", sep = ",")
head(data)

# looking at the structure of the dataset
str(data)

colnames(data)

#get a summary of the dataset, including measures of central tendency

summary(data)

#measures of dispersion
var(data)
sd(as.numeric(unlist(data)))

#scatter plot matrix
library(ggplot2)
pairs(data, aes(colour = Species))

#create box plots to compare the distributions of each variable for each species

boxplot(data)

#check for duplicates and missing values
library(naniar)
gg_miss_var(data)

sum(duplicated(data))

# Plot dependent variable distribution
table(data$credit_risk)

barplot(table(data$credit_risk))

attach(data)


# Standardize the data

data$amount = scale(amount)
data$age = scale(age)
data$duration = scale(duration)

boxplot(data)

# Correlation matrix
df <- data.frame(status, duration, credit_history, purpose, amount, savings,                
                  employment_duration, installment_rate, personal_status_sex, other_debtors,
                  present_residence,property,               
                  age,other_installment_plans,housing,number_credits,job, people_liable,          
                  telephone,foreign_worker,credit_risk )

df_cor <- cor(df, method = "spearman")
round(df_cor, digits = 2)

cor.df <- as.data.frame(df_cor)


#CorrPlot
library(corrgram)
library(corrplot)

corrplot(corr = cor(df))

str(data)


#FEATURE SELECTION
library(Boruta)
library(mlbench)

set.seed(123)
boruta = Boruta(credit_risk~., data = df, doTrace = 2)
print(boruta)
plot(boruta, las= 2, cex.axis = 0.7)
fs = TentativeRoughFix(boruta)
print(fs)
plot(fs, las= 2, cex.axis = 0.7)
getConfirmedFormula(fs)
attStats(fs)

# randomise order of the data
set.seed(12345)
df<- df[order(runif(1000)), ]

# create train and test data sets
# split data set
dftrain <- df[1:700, ]     # 70%
dftest <- df[701:1000, ]   # 30%
c_h <- dftest$credit_risk
dftest <- dftest[-21]


# Plot dependent variable distribution in the train dataset
table(dftrain$credit_risk)

barplot(table(dftrain$credit_risk))

#Model Balancing
library(ROSE)
library(smotefamily)

set.seed(123)

over = ovun.sample(credit_risk~., data = dftrain, method = "over")$data

table(over$credit_risk)

barplot(table(over$credit_risk))


#Logistic regression

# First round use all variables
lgr1 = glm(credit_risk ~ status + duration + credit_history + purpose + 
             amount + savings + employment_duration + installment_rate + 
             other_debtors + property + age + other_installment_plans + 
             housing + job, data = over, family = "binomial")
summary(lgr1)

# Calculate Odds Ratio - Exp(b) with 95% confidence intervals (2 tail)
exp(cbind(OR = coef(lgr1), confint(lgr1)))


# Second round excluding non-significant variables
lgr2 = glm(credit_risk ~ status + credit_history + 
             amount + savings + installment_rate + 
             other_debtors + property + age + housing + job, data = over, family = "binomial")
summary(lgr2)


#Variance Inflation factor

vif(lgr2)
sqrt(vif(lgr2)) > 2  # if > 2 vif too high


# Calculate Odds Ratio - Exp(b) with 95% confidence intervals (2 tail)
exp(cbind(OR = coef(lgr2), confint(lgr2)))

#predict with model 2

dftest <- dftest[c("status", "credit_history", "amount", "savings",                
            "installment_rate", "other_debtors", "property", 
            "age", "housing", "job")]
dfpred <- predict.glm(lgr2, dftest)
summary(dfpred)
dfpred <- ifelse(exp(dfpred) > 0.5, 1, 0)
dfpred <- as.factor(dfpred)
c_h <- as.factor(c_h)
# Assess accuracy
library(gmodels)
CrossTable(x = c_h, y = dfpred, prop.chisq = FALSE)

library(caret)
confusionMatrix(dfpred, c_h, positive = "1")


# Calculate accuracy, precision, and recall
#accuracy <- confusionMatrix(table(y_true, y_pred))$overall["Accuracy"]
#precision <- confusionMatrix(table(y_true, y_pred))$byClass["Pos Pred Value"]
#recall <- confusionMatrix(table(y_true, y_pred))$byClass["Sensitivity"]

# Calculate F1 score
#F1_score <- 2 * (precision * recall) / (precision + recall)
F1_score <- 2 * (0.8299 * 0.9050) / (0.8299 + 0.9050)
F1_score


# Convert factor variable to numeric
dfpred <- as.numeric(as.character(dfpred))

# Generate ROC curve
roc_obj <- roc(c_h, dfpred)

# Plot the ROC curve
plot(roc_obj, main = "ROC curve for logistic regression model", col = "blue", lwd = 2, print.auc = TRUE, legacy.axes = TRUE)

# Calculate AUC
auc_obj <- auc(roc_obj)
print(paste("AUC: ", auc_obj))


detach(data)


#DESCISION TREE

# training a model on the data
# build the simplest decision tree

library(C50)

over$credit_risk = factor (over$credit_risk)

str(over)

dtmodel <- C5.0(credit_risk ~ status + duration + credit_history + purpose + 
                  amount + savings + employment_duration + installment_rate + 
                  other_debtors + property + age + other_installment_plans + 
                  housing + job, data = over)

# display simple facts about the tree
dtmodel

# display detailed information about the tree
summary(dtmodel)

# evaluating model performance
attach(data)
dftest <- df[701:1000, ]   # 30%

dftest <- dftest[c("status","duration", "credit_history", "amount", "savings", "purpose",               
                   "employment_duration", "installment_rate", "other_debtors", "property", "age",
                   "other_installment_plans", "housing", "job")]

# create a factor vector of predictions on test data
dtpred <- predict(dtmodel, dftest)

dtpred
c_h
# cross tabulation of predicted versus actual classes
library(gmodels)

CrossTable(c_h, dtpred,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual credit risk', 'predicted credit risk'))

# more diagnostics
library(caret)
confusionMatrix(dtpred, c_h, positive = "1")


# Calculate F1 score
#F1_score <- 2 * (precision * recall) / (precision + recall)
F1_score <- 2 * (0.8364 * 0.8100) / (0.8364 + 0.8100)
F1_score


# Convert factor variable to numeric
dtpred <- as.numeric(as.character(dtpred))

# Generate ROC curve
roc_obj <- roc(c_h, dtpred)

# Plot the ROC curve
plot(roc_obj, main = "ROC curve for decision tree model", col = "blue", lwd = 2, print.auc = TRUE, legacy.axes = TRUE)

# Calculate AUC
auc_obj <- auc(roc_obj)
print(paste("AUC: ", auc_obj))

detach(data)



#SUPPORT VECTOR MACHINE

library(MASS)
library(DMwR2)
library(kernlab)

# run model
set.seed(12345)
svm <- ksvm(credit_risk ~ status + duration + credit_history + purpose + 
              amount + savings + employment_duration + installment_rate + 
              other_debtors + property + age + other_installment_plans + 
              housing + job, data = over,  kernel = "rbfdot", type = "C-svc")
# rbfdot is a Linear kernel; -- WARNING -- some kernels take a long time

# look at basic information about the model
svm

# Predict
svmpred <- predict(svm, dftest)

#Evaluate
table(svmpred, c_h)

# sum diagonal for accuracy
sum(diag(round(prop.table(table(svmpred, c_h))*100,1)))

c_h = factor(c_h)
svmpred = factor(svmpred)

# cross tabulation of predicted versus actual classes

CrossTable(c_h, svmpred,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual credit risk', 'predicted credit risk'))

confusionMatrix(svmpred, c_h, positive = "1")


# Calculate F1 score
#F1_score <- 2 * (precision * recall) / (precision + recall)
F1_score <- 2 * (0.8810 * 0.8371) / (0.8810 + 0.8371)
F1_score


# Convert factor variable to numeric
svmpred <- as.numeric(as.character(svmpred))

# Generate ROC curve
roc_obj <- roc(c_h, svmpred)

# Plot the ROC curve
plot(roc_obj, main = "ROC curve for support vector machine model", col = "blue", lwd = 2, print.auc = TRUE, legacy.axes = TRUE)

# Calculate AUC
auc_obj <- auc(roc_obj)
print(paste("AUC: ", auc_obj))


detach(data)



#RANDOM FOREST

# run model
# load the randomForest package
library(randomForest)
library(gmodels)
library(caret)

over$credit_risk = as.factor(over$credit_risk)

# fit a random forest model to the training data
rf <- randomForest(credit_risk ~ status + duration + credit_history + purpose + 
                     amount + savings + employment_duration + installment_rate + 
                     other_debtors + property + age + other_installment_plans + 
                     housing + job, data = over, type = "classification")
rf

# make predictions on the testing data
rfpred <- predict(rf, newdata = dftest)

table(rfpred, c_h)

# sum diagonal for accuracy
sum(diag(round(prop.table(table(rfpred, c_h))*100,1)))

# cross tabulation of predicted versus actual classes

CrossTable(c_h, rfpred,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual credit risk', 'predicted credit risk'))


confusionMatrix(rfpred, c_h, positive = "1")


# Calculate F1 score
#F1_score <- 2 * (precision * recall) / (precision + recall)
F1_score <- 2 * (0.8397 * 0.90) / (0.8397 + 0.90)
F1_score


# Convert factor variable to numeric
rfpred <- as.numeric(as.character(rfpred))

# Generate ROC curve
roc_obj <- roc(c_h, rfpred)

# Plot ROC curve
# Plot the ROC curve
plot(roc_obj, main = "ROC curve for random forest model", col = "blue", lwd = 2, print.auc = TRUE, legacy.axes = TRUE)


# Calculate AUC
auc_obj <- auc(roc_obj)
print(paste("AUC: ", auc_obj))

detach(data)


#MODEL OPTIMIZATION


# load packages
library(e1071)
library(caret)

#Create a grid of hyperparameters to search over
tuneGrid <- expand.grid(C = c(0.01, 0.1, 1, 10, 100),
                        sigma = c(0.01, 0.1, 1, 10))
set.seed(123)
#Perform cross-validation to tune the hyperparameters
svmModel <- train(credit_risk ~ status + duration + credit_history + purpose + 
                    amount + savings + employment_duration + installment_rate + 
                    other_debtors + property + age + other_installment_plans + 
                    housing + job, data = over, 
                  method = "svmRadial", 
                  tuneGrid = tuneGrid, 
                  trControl = trainControl(method = "cv", number = 5))

# Select the best hyperparameters
bestC <- svmModel$bestTune$C
bestSigma <- svmModel$bestTune$sigma



# Train the SVM model using the best hyperparameters
finalModel <- svm(credit_risk ~ status + duration + credit_history + purpose + 
                    amount + savings + employment_duration + installment_rate + 
                    other_debtors + property + age + other_installment_plans + 
                    housing + job, 
                  data = over, 
                  method = "svmRadial", 
                  cost = bestC, 
                  gamma = 1/(2*bestSigma^2))

# Test the model on the testing set
predictions <- predict(finalModel, dftest)

# sum diagonal for accuracy
sum(diag(round(prop.table(table(predictions, c_h))*100,1)))

# cross tabulation of predicted versus actual classes

CrossTable(c_h, predictions,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual credit risk', 'predicted credit risk'))


confusionMatrix(predictions, c_h, positive = "1")


# Calculate F1 score
#F1_score <- 2 * (precision * recall) / (precision + recall)
F1_score <- 2 * (0.9096 * 0.7738) / (0.9096 + 0.7738)
F1_score


# Convert factor variable to numeric
predictions <- as.numeric(as.character(predictions))

# Generate ROC curve
roc_obj <- roc(c_h, predictions)

# Plot ROC curve
# Plot the ROC curve
plot(roc_obj, main = "ROC curve for optimized SVM model", col = "blue", lwd = 2, print.auc = TRUE, legacy.axes = TRUE)

detach(data)
