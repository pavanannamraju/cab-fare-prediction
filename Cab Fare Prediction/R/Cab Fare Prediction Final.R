# Setting Working Directory
#########################################################################

setwd('D:/Edwisor Final Project/Final Project/R/')

#########################################################################


#########################################################################

# Reading Datasets


train <- read.csv('./datasets/train_cab.csv')
train

test <- read.csv('./datasets/test.csv')
test
#########################################################################

#########################################################################

# Data Understanding

str(train)
str(test)

summary(train)
summary(test)

#########################################################################

#########################################################################

# Missing Value analysis

table(is.na(train$fare_amount))

table(train$fare_amount == "")

train <- train[!(train$fare_amount == ""),]

table(is.na(train$passenger_count))

library(tidyverse)

train <- drop_na(train)
table(is.na(train))
str(train)

#########################################################################

#########################################################################

# Also removing some invalid values at a time since found before

train <- train[!(train$fare_amount == '430-'),]
nrow(train)

train <- train[!(train$pickup_datetime == '43'),]

table(train$passenger_count)
train <- train[(train$passenger_count != 0 ),]
train <- train[(train$passenger_count != 0.12 ),]
train <- train[(train$passenger_count<7),]

#########################################################################

#########################################################################

# Data Type conversion

train$fare_amount = as.numeric(train$fare_amount)
train$passenger_count <- as.integer(train$passenger_count)


#########################################################################

#########################################################################

# Feature Engineering

train$Year <- as.numeric(substr(train$pickup_datetime, 1, 4))
train$Month <- as.numeric(substr(train$pickup_datetime, 6, 7))
train$Day <- as.numeric(substr(train$pickup_datetime, 9, 10))
train$Hour <- as.numeric(substr(train$pickup_datetime, 12, 13))
train$Minute <- as.numeric(substr(train$pickup_datetime, 15,16))

test$Year <- as.numeric(substr(test$pickup_datetime, 1, 4))
test$Month <- as.numeric(substr(test$pickup_datetime, 6, 7))
test$Day <- as.numeric(substr(test$pickup_datetime, 9, 10))
test$Hour <- as.numeric(substr(test$pickup_datetime, 12, 13))
test$Minute <- as.numeric(substr(test$pickup_datetime, 15,16))


train[train$pickup_longitude > 180,]
train[train$pickup_longitude < -180,]

train[train$pickup_latitude > 90,]
train <- train[train$pickup_latitude < 90,]

train[train$pickup_latitude < -90,]


train[train$dropoff_longitude >180 ,]
train[train$dropoff_longitude < -180 ,]

train[train$dropoff_latitude > 90,]
train[train$dropoff_latitude < -90,]




require(geosphere)

train$dist <- distHaversine(cbind(train$pickup_longitude, train$pickup_latitude), cbind(train$dropoff_longitude, train$dropoff_latitude))
train$dist <- train$dist/1000

test$dist <- distHaversine(cbind(test$pickup_longitude, test$pickup_latitude), cbind(test$dropoff_longitude, test$dropoff_latitude))
test$dist = test$dist / 1000


# Removing pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude as it is no longer needed

train_new <- subset(train, select = -c(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude))
str(train_new)

test_new <- subset(test, select = -c(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude))
str(test_new)

#########################################################################

#########################################################################

# Taking backup of new data

train_old <- train
test_old <- test

train = train_new
test = test_new


# Removing pickup_datetime as it is no longer needed

train <- subset(train, select = -c(pickup_datetime))
test <- subset(test, select = -c(pickup_datetime))

str(train)
str(test)

#########################################################################

#########################################################################

# Sanity Checks

train <- train[train$fare_amount > 0,]

train <- train[train$dist > 0,]

test <- test[test$dist !=0,]


#########################################################################

#########################################################################

# Outlier Analysis

require(ggplot2)
require(car)
qqPlot(train$fare_amount)
boxplot(train$fare_amount)
hist(train$fare_amount)

# IQR()

q25 = quantile(train$fare_amount,0.25)
q75 = quantile(train$fare_amount,0.75)
iqr = q75 - q25

q25
q75
iqr
lower_limit =  q25 - (iqr * 1.5)
upper_limit = q75 + (iqr * 1.5)
lower_limit
upper_limit


train$fare_amount <- ifelse(train$fare_amount > upper_limit, upper_limit, train$fare_amount)


qqPlot(train$fare_amount)
boxplot(train$fare_amount)
hist(train$fare_amount)



qqPlot(train$dist)
boxplot(train$dist)
hist(train$dist)

q25 = quantile(train$dist,0.25)
q75 = quantile(train$dist,0.75)
iqr = q75 - q25

q25
q75
iqr
lower_limit =  q25 - (iqr * 1.5)
upper_limit = q75 + (iqr * 1.5)
lower_limit
upper_limit

summary(train$dist)

train$dist <- ifelse(train$dist > upper_limit, upper_limit, train$dist)

qqPlot(train$dist)
boxplot(train$dist)
hist(train$dist)


#########################################################################

#########################################################################

# Resetting Index of data

train.names <- NULL
test.names <- NULL

#########################################################################

#########################################################################

# Plotting between variables to understand the relationship between variables

plot(train$fare_amount, train$dist)

year    <- as.factor(train$Year)
month   <- as.factor(train$Month)
day     <- as.factor(train$Day)
hour    <- as.factor(train$Hour)
minute  <- as.factor(train$Minute)
pc <- as.factor(train$passenger_count)


ggplot(train, aes(x= year, y=fare_amount)) + 
  geom_violin()

ggplot(train, aes(x = month, y=fare_amount)) +
  geom_violin()


ggplot(train, aes(x = day, y=fare_amount)) +
  geom_violin()

ggplot(train, aes(x = pc, y=fare_amount)) +
  geom_violin()

#########################################################################

#########################################################################

# Model Development

# Splitting data into training and validation data

indices <- sample(1:nrow(train), size = 0.8 * nrow(train))
train1 <- train[indices,]
test1 <- train[-indices,]

# Defining r2_score metric for calculating accuracy

r2score <- function(actual, preds){
  rss <- sum((preds - actual) ^ 2)  ## residual sum of squares
  tss <- sum((actual - mean(actual)) ^ 2)  ## total sum of squares
  rsq <- 1 - rss/tss
  return (rsq)
}

# Linear Regression

linear_model= lm(fare_amount ~. ,  data= train1)
summary(linear_model)
pred = predict(linear_model, test1[,-1])
r2score(test1[,1],pred)


# Decision Tree Regression

# install.packages('rpart')
require(rpart)

dt = rpart(fare_amount ~. ,  data= train1,)
summary(dt)
pred = predict(dt, test1[,-1])
r2score(test1[,1],pred)

# Random Forest

# install.packages('randomForest')
require(randomForest)
rf <- randomForest(fare_amount ~., data = train1)
summary(rf)
pred = predict(rf, test1[,-1])
r2score(test1[,1],pred)

#  Gradient boosting

# install.packages('caret')
# install.packages('xgboost')
library(caret)
library(xgboost)

model <- train(
  fare_amount ~., data = train1, method = "xgbTree",
  trControl = trainControl("cv", number = 10)
)
model$bestTune
predictions <- model %>% predict(test1[,-1])
head(predictions)
r2score(test1[,1], predictions)

# From the above models, Gradient Boosting Regressor is the best accuracy providing model

#########################################################################

# Predicting data from test dataset.
test$fare_amount <- 0 
test <- test[,c(1:7)]

model <- train(
  fare_amount ~., data = train, method = "xgbTree",
  trControl = trainControl("cv", number = 10)
)
model$bestTune

test$fare_amount <- model %>% predict(test)

write.csv(test, 'D:/Edwisor Final Project/final_predictions_r.csv', row.names = F)

#########################################################################
