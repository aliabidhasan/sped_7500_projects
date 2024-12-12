library(tidyverse)

nlsy <- read_csv("NLSY_F.csv")

nlsy <- nlsy[,-1]

str(nlsy)
colSums(is.na(nlsy))
nlsy$HdSt_R4531700 <- ifelse(nlsy$HdSt_R4531700 == "Did Not Attend Head Start", 0, 
                             ifelse(nlsy$HdSt_R4531700 == "Attended Head Start",1,nlsy$HdSt_R4531700))

# pull interesting variables 
nlsy_sub <- nlsy %>% 
  dplyr::select(HGCm_R0006500,HGCf_R0007900, NumBks_R0017358, PctDis_R0017366, PctDO_R0017368,FTCouns_R0017381, 
         FTFac_R0017382, PCTFTGradDeg_R0017383, HdSt_R4531700, SelfInc1999_R6909701, SelfInc2009_T3045300,
         GPA, avg_IQ_percentile) %>% 
  filter(complete.cases(HGCm_R0006500,HGCf_R0007900, NumBks_R0017358, PctDis_R0017366, PctDO_R0017368,FTCouns_R0017381, 
                        FTFac_R0017382, PCTFTGradDeg_R0017383, HdSt_R4531700, SelfInc1999_R6909701, SelfInc2009_T3045300,
                        GPA, avg_IQ_percentile)) %>% 
  mutate(HdSt_R4531700 = as.numeric(HdSt_R4531700))

nrow(nlsy_sub) 
dim(nlsy_sub)
str(nlsy_sub)
# use LOOCV and say because small dataset because wanted to use IQ
# for mine do college and above or less than college
# correlation matrix
cor(nlsy_sub)

# nicer correlation matrix
library(corrplot)
corrplot(cor(nlsy_sub), 
         type="lower", #put color strength on bottom
         tl.pos = "ld", #Character or logical, position of text labels, 
         #'ld'(default if type=='lower') means left and diagonal,
         tl.cex = 1, #Numeric, for the size of text label (variable names).
         method="color", 
         addCoef.col="darkgray", 
         diag=FALSE,
         tl.col="black", #The color of text label.
         tl.srt=45, #Numeric, for text label string rotation in degrees, see text
         is.corr = FALSE, #if you include correlation matrix
         #order = "hclust", #order results by strength
         #col=gray.colors(100), #in case you want it in gray...
         number.digits = 3) #number of digits after decimal


library(PerformanceAnalytics) #for chart.Correlation

chart.Correlation(nlsy_sub, histogram = TRUE, method = "pearson")


# let's do linear regression 
library(car) 

# Predicting income in 2009
nlsy_reg_income <- lm(SelfInc2009_T3045300 ~ NumBks_R0017358 + PctDO_R0017368 + PctDis_R0017366 + 
                 FTCouns_R0017381 +  FTFac_R0017382 + GPA + SelfInc1999_R6909701 +
                 avg_IQ_percentile, nlsy_sub)
summary(nlsy_reg_income)

#RMSE
sqrt(mean(nlsy_reg_income$residuals^2))
# multicollinearity 
car::vif(nlsy_reg_income) # FT faculty is redundant with FT counselors

# remove FT faculty
nlsy_reg_income2 <- lm(SelfInc2009_T3045300 ~ NumBks_R0017358 + PctDO_R0017368 + PctDis_R0017366 + 
                        FTCouns_R0017381  + GPA + SelfInc1999_R6909701 +
                        avg_IQ_percentile, nlsy_sub)
summary(nlsy_reg_income2)

# multicollinearity 
car::vif(nlsy_reg_income2)

# Predicting HS GPA
nlsy_reg_gpa <- lm(GPA ~ NumBks_R0017358 + PctDO_R0017368 + PctDis_R0017366 + 
                     FTCouns_R0017381 + avg_IQ_percentile, nlsy_sub)
summary(nlsy_reg_gpa)

# multicollinearity 
car::vif(nlsy_reg_gpa) 

# graph predicted vs. fitted for GPA
library(ggplot2)
actual <- nlsy_sub$GPA
fitted <- unname(nlsy_reg_gpa$fitted.values) #would have been a named number vector if unname not used
#grab up the fitted values from the regression model

act_fit <- cbind.data.frame(actual, fitted) #cbind binds the two vectors into a dataframe


ggplot(act_fit, aes(x = actual, y = fitted)) +
  geom_point() +
  xlab("Actual value") +
  ylab("Predicted value") +
  ggtitle("Scatterplot for actual and fitted values") +
  geom_abline(intercept = 1,
              slope = 1,
              color = "maroon",
              linewidth = 1)

# time for some machine learning
library(caret)
# create training and test set
set.seed(1256) #initialize a pseudorandom number generator so that train and test set will be the same each time you call functions

#create new variable in tibble for division into training and test sets
nlsy_sub <- nlsy_sub %>% 
  mutate(id = row_number())

#70% of data as training set 
train_set <- nlsy_sub %>% 
  sample_frac(0.70) #which to select (the 70%)

#30% of data test set 
test_set  <- anti_join(nlsy_sub, train_set, by = 'id') 
#anti_join, basically says grab what is in final_tib that is not in train_set

#remove unnecessary variables 
train_set <- train_set %>% 
  dplyr::select(-id)

test_set <- test_set %>% 
  dplyr::select(-id)

# run model from above on the train set
lm_model <- train(GPA ~ NumBks_R0017358 + PctDO_R0017368 + PctDis_R0017366 + 
                    FTCouns_R0017381 + avg_IQ_percentile, 
                  data = train_set, #the data
                  method = "lm") #the method (lm = linear model)

summary(lm_model)

# lets remove percent disadvantaged at the school
lm_model <- train(GPA ~ NumBks_R0017358 + PctDO_R0017368 + 
                    FTCouns_R0017381 + avg_IQ_percentile, 
                  data = train_set, #the data
                  method = "lm") #the method (lm = linear model)

summary(lm_model)

# and remove number of books in school library
lm_model2 <- train(GPA ~PctDO_R0017368 + FTCouns_R0017381 + avg_IQ_percentile, 
                  data = train_set, #the data
                  method = "lm") #the method (lm = linear model)

summary(lm_model2)

# scatterplot?
#need fitted values from training set
fitted_train <- predict(lm_model2)

actual_train <- train_set$GPA
fitted_train <- unname(fitted_train) #would have been a named number vector if unname not used

act_fit_train <- cbind.data.frame(actual_train, fitted_train) #cbind binds the two vectors into a dataframe


ggplot(act_fit_train, aes(x = actual_train, y = fitted_train)) +
  geom_point() +
  xlab("Actual value") +
  ylab("Predicted value") +
  ggtitle("Scatterplot for actual and fitted values: Training data") +
  geom_abline(color = "maroon",
              linewidth = 1)

# predict test set:
lm_predict <- predict(lm_model2, test_set) #this will give us the predictions

#lm_predict

corr_test <- cor(lm_predict, test_set$GPA) #this give us the correlation between the actual score for percent_on_mastered in the test set and the predicted scores

corr_test

corr_test^2 #square for comparison to R^2 in OG model (slightly higher than our model)

# compare actual vs. fitted in test
actual_test <- test_set$GPA

act_fit_test <- cbind.data.frame(actual_test, lm_predict) #cbind binds the two vectors into a dataframe


ggplot(act_fit_test, aes(x = actual_test, y = lm_predict)) +
  geom_point() +
  xlab("Actual value") +
  ylab("Predicted value") +
  ggtitle("Scatterplot for actual and fitted values: Test data") +
  geom_abline(intercept = 1,
              slope = 1,
              color = "gold",
              linewidth = 1)


# 10-fold CV
library(relaimpo) #variable importance (it's new)

#set seed for replication of cross-validation at later time
set.seed(123)

# Set up repeated k-fold cross-validation
train.control <- trainControl(method = "cv", number = 10)
#method = cross validation, number = ten times (10 fold cross-validation)

nlsy_sub <- nlsy_sub %>% 
  dplyr::select(-id)

#the LM model used
lm_cv10 <- train(GPA ~ NumBks_R0017358 + PctDO_R0017368 + PctDis_R0017366 + 
                   FTCouns_R0017381 + avg_IQ_percentile, 
                 data = nlsy_sub,
                 method = "lm", 
                 trControl = train.control)


summary(lm_cv10)
varImp(lm_cv10)$importance

#select significant predictors and run again

lm_cv10_2 <- train(GPA ~ PctDO_R0017368 + 
                     FTCouns_R0017381 + avg_IQ_percentile, 
                   data = nlsy_sub,
                   method = "lm", 
                   trControl = train.control)

summary(lm_cv10_2)

varImp(lm_cv10_2)$importance #how important are the variables selected?

# LOOCV
dim(nlsy_sub)
set.seed(123)

# Set up repeated k-fold cross-validation
train.control <- trainControl(method = "cv", number = 684)
#method = cross validation, number = LOOCV

#the LM model used
lm_cvLOOCV <- train(GPA ~ NumBks_R0017358 + PctDO_R0017368 + PctDis_R0017366 + 
                   FTCouns_R0017381 + avg_IQ_percentile, 
                 data = nlsy_sub,
                 method = "lm", 
                 trControl = train.control)


summary(lm_cvLOOCV)
varImp(lm_cvLOOCV)$importance

#select significant predictors and run again

lm_LOOCV_2 <- train(GPA ~ PctDO_R0017368 + 
                     FTCouns_R0017381 + avg_IQ_percentile, 
                   data = nlsy_sub,
                   method = "lm", 
                   trControl = train.control)

summary(lm_LOOCV_2)

varImp(lm_LOOCV_2)$importance #how important are the variables selected?


#### Class today, 10/29/2024
# let's do some feature selection 
train.control <- trainControl(method = "cv", number = 10)

lm_cv10_featureselection <- train(GPA ~ HGCm_R0006500 + HGCf_R0007900 + PCTFTGradDeg_R0017383 + HdSt_R4531700 +
                                    NumBks_R0017358 + PctDO_R0017368 + PctDis_R0017366 + 
                   FTCouns_R0017381 + FTFac_R0017382 +avg_IQ_percentile, 
                 data = nlsy_sub,
                 #method = "leapForward", #stepwise selection
                 #method = "leapBackward", #stepwise selection
                 method = "leapSeq", #stepwise selection 
                 trControl = train.control)
summary(lm_cv10_featureselection)
# regardless of stepwise method, all land on model with GPA ~ FTCouns + avg_IQ_percentile
# but did once more and now got it to pull Percent Dropout too

lm_gpa <- lm(GPA ~ HGCm_R0006500 + HGCf_R0007900 + PCTFTGradDeg_R0017383 + HdSt_R4531700 +
               NumBks_R0017358 + PctDO_R0017368 + PctDis_R0017366 + 
               FTCouns_R0017381 + FTFac_R0017382 +avg_IQ_percentile, 
             data = nlsy_sub)
summary(lm_gpa)

# let's try income 
lm_cv10_featureselection <- train(SelfInc2009_T3045300 ~ GPA+ HGCm_R0006500 + HGCf_R0007900 + PCTFTGradDeg_R0017383 + HdSt_R4531700 +
                                    NumBks_R0017358 + PctDO_R0017368 + PctDis_R0017366 + 
                                    FTCouns_R0017381 + FTFac_R0017382 +avg_IQ_percentile, 
                                  data = nlsy_sub,
                                  #method = "leapForward", #stepwise selection
                                  #method = "leapBackward", #stepwise selection
                                  method = "leapSeq", #stepwise selection 
                                  trControl = train.control)
summary(lm_cv10_featureselection)
# regardless of stepwise method, all land on model with GPA ~ FTCouns + avg_IQ_percentile
# but did once more and now got it to pull Percent Dropout too

lm_inc <- lm(SelfInc2009_T3045300 ~ GPA+ HGCm_R0006500 + HGCf_R0007900 + PCTFTGradDeg_R0017383 + HdSt_R4531700 +
               NumBks_R0017358 + PctDO_R0017368 + PctDis_R0017366 + 
               FTCouns_R0017381 + FTFac_R0017382 +avg_IQ_percentile, 
             data = nlsy_sub)
summary(lm_inc)
