data<-read.csv("C:/Users/vaish/Downloads//car_details.csv")
data

#Dimensions
dim(data)

#Delete car column
data <- data[, !(names(data) %in% c("name"))]
head(data)

#Missing values 
data <- na.omit(data)
#Correlation Plot 
library(corrplot)
numeric_columns <- sapply(data, is.numeric)
correlation_matrix <- cor(data[, numeric_columns])
corrplot(correlation_matrix, method = "number", tl.col = "red", tl.srt = 45)

##############Exploratory Data Analysis#############
columns_of_interest <- c("year", "selling_price", "km_driven"
                         ,"mileage","engine_cc","max_power","seats","torque")  # Replace with actual column names
par(mfrow=c(3,3)) #https://www.datamentor.io/r-programming/subplot
for(col in columns_of_interest) {
  if(is.numeric(data[[col]])) {
    hist(data[[col]], main = paste("Histogram of", col), xlab = col)
  }
}
for(col in columns_of_interest) {
  if(is.numeric(data[[col]])) {
    boxplot(data[[col]], main = paste("Boxplot of", col), xlab = col)
  }
}

trans <- preProcess(data, method = c("BoxCox"))  ## need {caret} package
trans

transformed <- predict(trans, data)

#Visualization for categorical variables.

my_data <- data.frame(
  encoded_fuel_type = sample(0:1, 8128, replace = TRUE)
)
par(mfrow=c(2, 2))
ggplot(my_data, aes(x = factor(encoded_fuel_type))) +
  geom_bar(fill = "skyblue", color = "black") +
  labs(title = "Distribution of Fuel Type")
##############
my_data <- data.frame(
  encoded_seller_type = sample(0:1, 8128, replace = TRUE)
)
ggplot(my_data, aes(x = factor(encoded_seller_type))) +
  geom_bar(fill = "skyblue", color = "black") +
  labs(title = "Distribution of Seller Type")
#################
my_data <- data.frame(
  encoded_owner = sample(0:1, 8128, replace = TRUE)
)
ggplot(my_data, aes(x = factor(encoded_owner))) +
  geom_bar(fill = "skyblue", color = "black") +
  labs(title = "Distribution of Type of Owner")
#############
my_data <- data.frame(
  encoded_transmission = sample(0:1, 8128, replace = TRUE)
)
ggplot(my_data, aes(x = factor(encoded_transmission))) +
  geom_bar(fill = "skyblue", color = "black") +
  labs(title = "Distribution of Type of Transmission")


#creating dummy variables
encoded_fuel_type <- model.matrix(~ fuel_type - 1, data)
encoded_seller_type <- model.matrix(~ seller_type - 1, data)
encoded_transmission <- model.matrix(~ transmission - 1, data)
encoded_owner <- model.matrix(~ owner - 1, data)

encoded_data <- cbind(data, encoded_fuel_type, encoded_seller_type, encoded_transmission, encoded_owner)

encoded_data <- subset(encoded_data, select = -c(fuel_type, seller_type, transmission, owner))
names(encoded_data)
head(encoded_data)

#delete x190 coloumn
encoded_data <- encoded_data[, !(names(encoded_data) %in% c("X190"))]
head(encoded_data)
dim(encoded_data)

#Near zero variables
library(caret)
near_var <- nearZeroVar(encoded_data)
df_final<-encoded_data[, -near_var]
dim(df_final)

#Highly Correlated
highCor<-findCorrelation(cor(df_final),cutoff = .80)
filtereddf <- df_final[,-highCor]
dim(filtereddf)

hist(selling_price)
selling_price
###########################################################################################################################################
#Splitting data 80% 20%
library(lars)
library(elasticnet)
set.seed(100)

trainIndex <- createDataPartition(filtereddf$selling_price, p = 0.8, 
                                  list = FALSE, 
                                  times = 1)
train_data <- filtereddf[ trainIndex,]
test_data  <- filtereddf[-trainIndex,]
test_target <- test_data$selling_price
# Define the training control for cross-validation
trainControl <- trainControl(method = "cv", number = 3)

##################LINEAR CONTINUOUS MODELS###############
#linear regression model
lm_model <- train(selling_price ~ ., data = train_data, method = "lm",metric = "RMSE",
                  trControl = trainControl)
lm_model
xyplot(train_data$selling_price ~ predict(lm_model), type = c("p", "g"), xlab = "Predicted", ylab = "Observed")
xyplot(resid(lm_model) ~ predict(lm_model), type = c("p", "g"), 
       xlab = "Predicted", ylab = "Residuals")

#  Ridge model
ridgeGrid <- data.frame(.lambda = seq(0, .1, length = 15))
ridge_model <- train(selling_price ~ ., data = train_data, method = "ridge",metric = "RMSE",
                     trControl = trainControl, tuneGrid = ridgeGrid,
                     preProc = c("center", "scale"))
ridge_model
plot(ridge_model)

# Lasso model
lassoGrid <- data.frame(alpha = 1, lambda = seq(0.001, 1, length = 100))
lasso_model <- train(selling_price ~ ., data = train_data, method = "glmnet",metric = "RMSE",
                     trControl = trainControl,
                     tuneGrid = lassoGrid,preProc = c("center", "scale"))
lasso_model
plot(lasso_model)

# Elastic Net model
enetGrid <- expand.grid(.lambda = c(0, 0.01, .1), .fraction = seq(.05, 1, length = 20))
elastic_net_model <- train(selling_price ~ ., data = train_data, method = "enet",
                           trControl = trainControl,
                           tuneGrid = enetGrid, preProc = c("center", "scale"))
elastic_net_model
plot(elastic_net_model)

#Predicting on test data

pred_enet<-predict(elastic_net_model, newdata = test_data)
pred_df4 <- data.frame(obs = test_target, pred = pred_enet)
defaultSummary(pred_df4)

###############################################################################################################################


###################NON-LINEAR CONTINUOUS MODELS####################
# Split the data into training and test sets (80%/20%)
library(caret)
trainIndex <- sample(nrow(filtereddf),nrow(filtereddf)*0.8)
train_pred <- filtereddf[ trainIndex, -1]
train_op <- filtereddf[trainIndex, 1]
test_pred  <- filtereddf[-trainIndex, -1]
test_op <- filtereddf[-trainIndex, 1]

dim(filtereddf)
# Define the training control for cross-validation
trainControl <- trainControl(method = "cv", number = 3)

######Neural Network######
nnetGrid <- expand.grid(.decay = c(0, 0.01, 0.1),
                        .size = c(1:5),
                        .bag = FALSE)

set.seed(100)
ctrl <- trainControl(method = "cv", number = 3)
nnetFit <- train(train_pred,train_op,
                 method = "avNNet",
                 tuneGrid = nnetGrid,
                 trControl = ctrl,
                 preProc = c("center", "scale"),
                 linout = TRUE,
                 trace = FALSE,
                 MaxNWts = 10 * (ncol(train_pred) + 1) + 10 + 1,
                 maxit = 100)
nnetFit
plot(nnetFit)
########SVM#######
svm_Grid <- expand.grid(sigma = c(0.01, 0.1, 1),
                      C = c(0.1, 1, 10))

svmRFit <- train(train_pred, train_op,
                 method = "svmRadial",
                 preProc = c("center", "scale"),
                 tuneGrid = svm_Grid,
                 trControl = trainControl(method = "cv", number = 3))

svmRFit

plot(svmRFit)
###########KNN#########
knnTune <- train(train_pred,train_op,
                 method = "knn",
                 # Center and scaling will occur for new predictions too
                 preProc = c("center", "scale"),
                 tuneGrid = data.frame(.k = 1:20),
                 trControl = trainControl(method = "cv", number = 3))
knnTune
plot(knnTune)
##########MARS######
mars_Grid <- expand.grid(nprune=seq(2:50),degree=seq(1:2))

#tune model
MARSModel <- train(train_pred,train_op,
                   method='earth',
                   tuneGrid=mars_Grid,
                   trControl = trainControl(method = "cv", number = 3))



MARSModel
mars_predictions <- predict(MARSModel,test_pred)
plot(MARSModel)

#Predicting on test data
svm_predictions <- predict(svmRFit,test_pred)
svm_Results <- postResample(pred = svm_predictions, obs = test_op)
svm_Results

################13.C###########################
################IMPORTANT VARIABLES##############
plsImpSim <- varImp(svmRFit, scale = FALSE)
plsImpSim

plot(plsImpSim, top = 5, scales = list(y = list(cex = .95)))
