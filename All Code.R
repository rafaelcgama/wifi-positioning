#### DELETES GLOBAL ENV VARIABLES ####
rm(list = ls())

#### SET WORKING DIRECTORY ####
# Uses the here package to resolve paths relative to the .Rproj file
# so this script works on any machine without editing paths.
# If you don't have 'here', run: install.packages("here")
library(here)
getwd()  # should point to the project root
set.seed(123)

#### LOAD LIBRARIES ####
library(caret)
library(readr)
library(dplyr)
library(tidyr)
library(data.table)
library(ggplot2)
library(RWeka)
library(ggthemes)
library(plotly)
library(e1071)

#Install required libraries:
#install.packages(c("here", "caret", "readr", "dplyr", "tidyr",
#                   "data.table", "ggplot2", "RWeka", "ggthemes", "plotly", "e1071"))

#+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=
#### LOAD DATA ####

#### TRAINING SET ####

#Load file — dataset must be placed in the data/ folder (see DATA.md)
training <- fread(here("data", "trainingData.csv"), sep = ",", stringsAsFactors = FALSE)

#converts to a tibble
training <- as_tibble(training)


#### VALIDATION SET ####

#Load file
validation <- fread(here("data", "validationData.csv"), sep = ",", stringsAsFactors = FALSE)

#converts to a tibble
validation <- as_tibble(validation)


#### Creates a partion to identify the dataset as training and validation to combine them ####
training$PARTITION <- "train"
validation$PARTITION <- "val"

combinedSets <- rbind(training, validation)



#+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=
#### DATA WRANGLING ####
#+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=

#Change data types to factors
combinedSets$FLOOR <- factor(combinedSets$FLOOR)
combinedSets$BUILDINGID <- factor(combinedSets$BUILDINGID)
combinedSets$SPACEID <- factor(combinedSets$SPACEID)
combinedSets$RELATIVEPOSITION <- factor(combinedSets$RELATIVEPOSITION)
combinedSets$USERID <- factor(combinedSets$USERID)
combinedSets$PHONEID <- factor(combinedSets$PHONEID)
combinedSets$PARTITION <- factor(combinedSets$PARTITION)



#### fuction to convert low signals to -100 ####
convertinglowsignals <- function(x) {
  if (x == 100) {
    x <- -100
  } else if (x <= -90) {
    x <- -100
  } else {
    x <- x
  }
}

#### Converting low signals WAPs to -100s and joining to the remaining portion of the dataset ####
combinedSets_waps <- as_tibble(apply(combinedSets[ , 1:((ncol(combinedSets) - 10))], c(1,2), convertinglowsignals))
combinedSets_other <- combinedSets[ , ((ncol(combinedSets) - 10) + 1):ncol(combinedSets)]
combinedSets <- cbind(combinedSets_waps, combinedSets_other)




#### REMOVING ROWS/COLS WITH ZERO VARIANCE

#### Function to Remove Rows ####
remove_no_signal_row <- function(data) {
  variances_row <- apply(data[ , 1:(ncol(data) - 10)], 1, var)
  data <-  data[which(variances_row != 0), ]
  return (data)
}


#### Function to Remove Cols ####
remove_no_signal_col <- function(data) {
  variances_col <- apply(data[ , 1:(ncol(data) - 10)], 2, var)
  data <- data[ , c(which(variances_col != 0), ((ncol(data) - 10) + 1):ncol(data))]
  return(data)
}



#### Remove all columns with no signal ####
combinedSets <- remove_no_signal_col(combinedSets)


#### Remove repetitive rows ####
combinedSets <- distinct(combinedSets)




#### CHECK HISTOGRAM FOR SIGNAL DISTRIBUTION ####

#### Creates a vector to check for unique readings ####
wap_values <- c(as.matrix(combinedSets[1:(ncol(combinedSets) - 10)]))
str(wap_values)
sort(unique(wap_values))
length(sort(unique(wap_values)))

#### Plot histogram of readings ####
ggplot() + geom_histogram(mapping = aes(x = wap_values[wap_values != -100]), bins = 30) +
  labs(x = "Readings", title = "Count of Readings")  + 
  theme(plot.title = element_text(hjust = 0.5))


#### CHECKING FOR PROBLEMS OF DATA VALUES > -30 ####

# Creates a vector to store indexes of rows that contain values > -30
index_above_30 <-  as.data.frame(which(combinedSets[, 1:(ncol(combinedSets) - 10)] > -30, arr.ind = TRUE))
index_above_30 <- unique(index_above_30$row)


#Creates a dataframe with the rows containing values > -30
data_above_30  <- combinedSets[index_above_30, ]


#Get names of WAPS that have values > -30 by row
badwaps <- colnames(data_above_30[1:(ncol(combinedSets) - 10)])[apply(data_above_30[1:(ncol(combinedSets) - 10)], 1, which.max)]


#Count the the number of times WAPs have values > -30
length(unique(badwaps)) # # there are 55 bad WAPS
sort(table(badwaps), decreasing = TRUE) # WAP087 has freq of 61, 
                                        # WAP065 has 35 so it is probably irrelevant when compared to the 20k obs


#Count how many values > -30 each user has
count(data_above_30, USERID)  # User #6 has 430 and user #14 has 51 occasions where the value is above -30


# count how many observations user 6 has
user6 <- combinedSets %>% 
  filter(USERID == 6)


nrow(user6) # 976 Obs 

#430 out of 976 observations for user 6 is bad - USER 6 should be removed




#### REMOVING USER 6 FROM DATA ####
combinedSets <- combinedSets %>% 
  filter(USERID != 6)


#### Transform values > -30 to -100 ####

combinedSets_waps <- as_tibble(apply(combinedSets[ , 1:((ncol(combinedSets) - 10))], c(1,2), function (x) ifelse(x > -30,-100,x)))
combinedSets_other <- combinedSets[ , ((ncol(combinedSets) - 10) + 1):ncol(combinedSets)]
combinedSets <- cbind(combinedSets_waps, combinedSets_other)




#### REMOVE ZERO VARIANCE ROWS AND COLS BEFORE NORMALIZING BY ROW ####

combinedSets <- remove_no_signal_col(combinedSets) # Has to be applied again to address the zero variance column 
                                                   # after normalizing by col that was caused by user 6 deletion 
combinedSets_byrow <- remove_no_signal_row(combinedSets)




#### NORMALIZATION ####

#### Divides in Wap portions of the data for different normalization methods ####
combinedSets_wap_all <- combinedSets[ , 1:(ncol(combinedSets) - 10)]
combinedSets_wap_byrow <- combinedSets_byrow[ , 1:(ncol(combinedSets_byrow) - 10)]
combinedSets_wap_bycol <- combinedSets[ , 1:(ncol(combinedSets) - 10)]
combinedSets_other <- combinedSets[ , ((ncol(combinedSets) - 10) + 1):ncol(combinedSets)]



#### Normalize function for all waps ####
normalize_all <- function(x) {
  num = x - min(combinedSets_wap_all)
  denom = max(combinedSets_wap_all) - min(combinedSets_wap_all)
  return (num/denom)
}



#### Normalize function by row/col ####
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}


##### Normalize using different methods ####
combinedSets_wap_normalized_all <- as_tibble(apply(combinedSets_wap_all, 2, normalize_all))
combinedSets_wap_normalized_byrow <- as_tibble(t(apply(combinedSets_wap_byrow, 1, normalize))) 
combinedSets_wap_normalized_bycol <- as_tibble(apply(combinedSets_wap_bycol, 2, normalize)) 


# Checks for NAs in the datasets
sum(!complete.cases(combinedSets_wap_normalized_all))
sum(!complete.cases(combinedSets_wap_normalized_byrow))
sum(!complete.cases(combinedSets_wap_normalized_bycol))


#Fix combinedSets_other to have the same indexes as combinedSets_wap_normalized_byrow
nrow(combinedSets_wap_byrow)
index <- c(rownames(combinedSets_wap_byrow))
combinedSets_other_by_row <- combinedSets_other[index, ]


#### Creates normalized full datasets ####
combinedSets_normalized_all <- cbind(combinedSets_wap_normalized_all, combinedSets_other)
combinedSets_normalized_byrow <- cbind(combinedSets_wap_normalized_byrow, combinedSets_other_by_row)
combinedSets_normalized_bycol <- cbind(combinedSets_wap_normalized_bycol, combinedSets_other)


#### Creates normalized wap values for viewing ####
wap_values_norm_all <- c(as.matrix(combinedSets_wap_normalized_all))
wap_values_nrom_byrow <- c(as.matrix(combinedSets_wap_normalized_byrow))
wap_values_nrom_bycol <- c(as.matrix(combinedSets_wap_normalized_bycol))


#### Histogram of normalized results ####
# ALL
ggplot() + 
  geom_histogram(aes(x = wap_values_norm_all[wap_values_norm_all != 0]), bins = 30) +
  labs(x = "Readings", title = "Count of Readings")  + 
  theme(plot.title = element_text(hjust = 0.5))


#BY ROW
ggplot() +
  geom_histogram(aes(x = wap_values_nrom_byrow[wap_values_nrom_byrow != 0]), bins = 30) +
  labs(x = "Readings", title = "Count of Readings Normalized by Row")  +
  theme(plot.title = element_text(hjust = 0.5))

#BY COL
ggplot() + 
  geom_histogram(aes(x = wap_values_nrom_bycol[wap_values_nrom_bycol != 0]), bins = 30) +
  labs(x = "Readings", title = "Count of Readings Normalized by Column")  + 
  theme(plot.title = element_text(hjust = 0.5))





#### DIVIDE DATASET IN TRAINING AND VALIDATION ####

# Choose from the following normalized datasets
# combinedSets_normalized_all 
# combinedSets_normalized_byrow 
# combinedSets_normalized_bycol


# Selected data normalized by row because it is the wants the yields the best predictions
training <- combinedSets_normalized_byrow %>% 
  filter(PARTITION == "train")


validation <- combinedSets_normalized_byrow %>% 
  filter(PARTITION == "val")



#### Sample training dataset to be used in training/testing/validation ####
set.seed(123)

#Samples rows
data_index_sampled <- sample(1:nrow(training), 3000)

#Creates sampled dataset 
training_sampled <- training[data_index_sampled, ]








#### Plot of Floors per building ####
ggplot(data = training_sampled, aes(x = BUILDINGID, y = FLOOR)) + 
  geom_point()

#### Plot of Longitude per Building ####
ggplot(data = training_sampled, aes(x = BUILDINGID, y = LONGITUDE)) + 
  geom_point()









#+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=
#### BUILDING ID ####
#+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=

set.seed(123)
##### Set the k-fold cross validation ####
ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 3
                     )

                     
#### TRAINING BUILDING ID ####

building_index <- grep("BUILDINGID", colnames(training_sampled))

#### Training partition for Building ####
training_sampled_Building <- training_sampled[ ,c(1:(ncol(training_sampled) - 10), building_index)]


#### SVM3 ####
training_svm3_building <- train(BUILDINGID ~ . ,
                                data = training_sampled_Building,      # train data partition
                                method = "svmLinear3",                 # type of model
                                preProc = c("center", "scale"),        # normalization
                                tuneLength = 10,                       # how many Ks
                                trControl = ctrl,
                                metric = "Accuracy"
                                
)

training_svm3_building
#Test Results:
# cost    Loss  Accuracy   Kappa    
# 0.25  L1    0.9998885  0.9998268



#### VALIDATION BUILDING ID ####

# Removes columns in the validation set to match training set
building_validation <- validation[ , which(names(validation) %in% names(training_sampled))]


# Predicts validation set using SVM3
svm3_validation_building <- predict(training_svm3_building, building_validation)

postResample(svm3_validation_building, building_validation$BUILDINGID)
# Accuracy    Kappa 
# 1           1





#+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=
#### FLOOR ####
#+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=


#### TRAINING FLOOR ####

floor_index <- grep("FLOOR", colnames(training_sampled))

#### Training partition for Floor ####
training_sampled_floor <- training_sampled[ ,c(1:(ncol(training_sampled) - 10), floor_index)]


#### SVM ####
training_svm_floor <- train(FLOOR ~ . ,
                                        data = training_sampled_floor,          # train data partition
                                        method = "svmLinear",                   # type of model
                                        preProc = c("center", "scale"),         # normalization
                                        tuneLength = 10,                        # how many Ks
                                        trControl = ctrl,
                                        metric = "Accuracy"
                                        
)

training_svm_floor
# Accuracy   Kappa    
# 0.9890018  0.9854729




#### VALIDATION FLOOR ####

floor_validation <- validation

# Substitute BuildingID data by the predicted BuildingID
floor_validation$BUILDINGID <- svm3_validation_building


# Removes columns in the validation set to match training set
floor_validation <- floor_validation[ , which(names(floor_validation) %in% names(training_sampled_floor))]


# Predicts validation set using SVM
svm_validation_floor <- predict(training_svm_floor, floor_validation)

postResample(svm_validation_floor, floor_validation$FLOOR)
# Accuracy    Kappa 
# 0.883679    0.837570




#+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=
#### FLOOR - BUILDING 0 ####
#+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=


#### TRAINING FLOOR - BUILDING 0 ####

# Group dataset by building
training_sampled_B0 <- training_sampled %>% 
  filter(BUILDINGID == 0)

# Refactor Floor because there are 5 levels(4 floors) but building 0 and 1 have only 4 floors 
training_sampled_B0$FLOOR <- factor(training_sampled_B0$FLOOR)


#### Training partition for Floor ####
training_sampled_floor_B0 <- training_sampled_B0[ ,c(1:(ncol(training_sampled_B0) - 10), floor_index)]



#### SVM ####
training_svm_floor_B0 <- train(FLOOR ~ . ,
                               data = training_sampled_floor_B0,       # train data partition
                               method = "svmLinear",                   # type of model
                               preProc = c("center", "scale"),         # normalization
                               tuneLength = 10,                        # how many Ks
                               trControl = ctrl,
                               metric = "Accuracy"
                               
)

training_svm_floor_B0
# Accuracy   Kappa    
# 0.9845508  0.9793096





#### VALIDATION FLOOR - BUILDING 0 ####

floor_B0_validation <- validation

# Substitute BuildingID data by the predicted BuildingID
floor_B0_validation$BUILDINGID <- svm3_validation_building

# Selects only building 0 from validation dataset
floor_B0_validation <- floor_B0_validation %>% 
  filter(BUILDINGID == 0)

# Removes columns in the validation set to match training set
floor_B0_validation <- floor_B0_validation[ , which(names(floor_B0_validation) %in% names(training_sampled_floor_B0))]



# Predicts validation set using SVM
svm_validation_floor_B0 <- predict(training_svm_floor_B0, floor_B0_validation)

postResample(svm_validation_floor_B0, floor_B0_validation$FLOOR)
# Accuracy     Kappa 
# 0.9440299    0.9209455








#### TRAINING FLOOR - BUILDING 1 ####

# Group dataset by building
training_sampled_B1 <- training_sampled %>% 
  filter(BUILDINGID == 1)

# Refactor Floor because there are 5 levels(4 floors) but building 1 and 1 have only 4 floors 
training_sampled_B1$FLOOR <- factor(training_sampled_B1$FLOOR)


#### Training partition for Floor ####
training_sampled_floor_B1 <- training_sampled_B1[ ,c(1:(ncol(training_sampled_B1) - 10), floor_index)]




#### SVM ####
training_svm_floor_B1 <- train(FLOOR ~ . ,
                               data = training_sampled_floor_B1,          # train data partition
                               method = "svmLinear",                   # type of model
                               preProc = c("center", "scale"),         # normalization
                               tuneLength = 10,                        # how many Ks
                               trControl = ctrl,
                               metric = "Accuracy"
                               
)

training_svm_floor_B1
# Accuracy   Kappa    
# 0.9914738  0.9884428



#### VALIDATION FLOOR - BUILDING 1 ####

floor_B1_validation <- validation

# Substitute BuildingID data by the predicted BuildingID
floor_B1_validation$BUILDINGID <- svm3_validation_building

# Selects only building 1 from validation dataset
floor_B1_validation <- floor_B1_validation %>% 
  filter(BUILDINGID == 1)

# Removes columns in the validation set to match training set
floor_B1_validation <- floor_B1_validation[ , which(names(floor_B1_validation) %in% names(training_sampled_floor_B1))]



# Predicts validation set using SVM
svm_validation_floor_B1 <- predict(training_svm_floor_B1, floor_B1_validation)

postResample(svm_validation_floor_B1, floor_B1_validation$FLOOR)
# Accuracy     Kappa 
# 0.7967213 0.7122598 






#### TRAINING FLOOR - BUILDING 2 ####

# Group dataset by building
training_sampled_B2 <- training_sampled %>% 
  filter(BUILDINGID == 2)


#### Training partition for Floor ####
training_sampled_floor_B2 <- training_sampled_B2[ ,c(1:(ncol(training_sampled_B2) - 10), floor_index)]




#### SVM ####
training_svm_floor_B2 <- train(FLOOR ~ . ,
                               data = training_sampled_floor_B2,       # train data partition
                               method = "svmLinear",                   # type of model
                               preProc = c("center", "scale"),         # normalization
                               tuneLength = 10,                        # how many Ks
                               trControl = ctrl,
                               metric = "Accuracy"
                               
)

training_svm_floor_B2
# Accuracy   Kappa    
# 0.9927322  0.9904665



#### VALIDATION FLOOR - BUILDING 2 ####

floor_B2_validation <- validation

# Substitute BuildingID data by the predicted BuildingID
floor_B2_validation$BUILDINGID <- svm3_validation_building

# Selects only building 2 from validation dataset
floor_B2_validation <- floor_B2_validation %>% 
  filter(BUILDINGID == 2)

# Removes columns in the validation set to match training set
floor_B2_validation <- floor_B2_validation[ , which(names(floor_B2_validation) %in% names(training_sampled_floor_B2))]



# Predicts validation set using SVM
svm_validation_floor_B2 <- predict(training_svm_floor_B2, floor_B2_validation)

postResample(svm_validation_floor_B2, floor_B2_validation$FLOOR)
# Accuracy     Kappa 
# 0.8656716 0.8165186










