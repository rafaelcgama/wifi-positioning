#### DELETES GLOBAL ENV VARIABLES ####
rm(list = ls())

#### LOAD LIBRARIES ####
library(here)       # Portable path resolution relative to .Rproj
library(caret)
library(readr)
library(dplyr)
library(tidyr)
library(data.table)
library(ggplot2)
library(RWeka)
library(ggthemes)
library(e1071)

set.seed(123)

# NOTE: Dataset CSVs are not included in this repository due to size.
# Download them from: https://archive.ics.uci.edu/ml/machine-learning-databases/00310/
# Place trainingData.csv and validationData.csv inside the data/ folder.

#+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=
#### LOAD DATA ####
#+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=

#### TRAINING SET ####
training <- fread(here("data", "trainingData.csv"), sep = ",", stringsAsFactors = FALSE)
training <- as_tibble(training)

#### VALIDATION SET ####
validation <- fread(here("data", "validationData.csv"), sep = ",", stringsAsFactors = FALSE)
validation <- as_tibble(validation)

# Tag each partition before combining
training$PARTITION   <- "train"
validation$PARTITION <- "val"

combinedSets <- rbind(training, validation)



#+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=
#### DATA WRANGLING ####
#+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=

# Change categorical columns to factors
combinedSets$FLOOR            <- factor(combinedSets$FLOOR)
combinedSets$BUILDINGID       <- factor(combinedSets$BUILDINGID)
combinedSets$SPACEID          <- factor(combinedSets$SPACEID)
combinedSets$RELATIVEPOSITION <- factor(combinedSets$RELATIVEPOSITION)
combinedSets$USERID           <- factor(combinedSets$USERID)
combinedSets$PHONEID          <- factor(combinedSets$PHONEID)
combinedSets$PARTITION        <- factor(combinedSets$PARTITION)


#### Convert low / undetected signals to -100 ####
# 100  = signal not detected
# <=-90 = very weak signal
# Both are mapped to -100 for consistency
convert_low_signals <- function(x) {
  ifelse(x == 100 | x <= -90, -100, x)
}

n_waps <- ncol(combinedSets) - 10   # Last 10 cols are metadata

combinedSets_waps  <- as_tibble(apply(combinedSets[, 1:n_waps], c(1, 2), convert_low_signals))
combinedSets_other <- combinedSets[, (n_waps + 1):ncol(combinedSets)]
combinedSets       <- cbind(combinedSets_waps, combinedSets_other)


#### Helper functions to remove zero-variance rows/cols ####
remove_no_signal_row <- function(data) {
  n <- ncol(data) - 10
  variances_row <- apply(data[, 1:n], 1, var)
  data[which(variances_row != 0), ]
}

remove_no_signal_col <- function(data) {
  n <- ncol(data) - 10
  variances_col <- apply(data[, 1:n], 2, var)
  data[, c(which(variances_col != 0), (n + 1):ncol(data))]
}

# Remove WAPs with no variance across all observations
combinedSets <- remove_no_signal_col(combinedSets)

# Remove duplicate rows
combinedSets <- distinct(combinedSets)


#### Exploratory: signal distribution ####
wap_values <- c(as.matrix(combinedSets[, 1:(ncol(combinedSets) - 10)]))
ggplot() +
  geom_histogram(aes(x = wap_values[wap_values != -100]), bins = 30) +
  labs(x = "Signal Strength (dBm)", title = "Distribution of Detected WAP Readings") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))


#### Investigate WAP readings above -30 dBm ####
# Values above -30 are atypically strong and likely measurement errors
index_above_30 <- as.data.frame(
  which(combinedSets[, 1:(ncol(combinedSets) - 10)] > -30, arr.ind = TRUE)
)
index_above_30 <- unique(index_above_30$row)
data_above_30  <- combinedSets[index_above_30, ]

# Which WAPs are the worst offenders?
badwaps <- colnames(data_above_30[, 1:(ncol(combinedSets) - 10)])[
  apply(data_above_30[, 1:(ncol(combinedSets) - 10)], 1, which.max)
]
sort(table(badwaps), decreasing = TRUE)  # WAP087 (61), WAP065 (35) are top offenders

# How many bad readings does each user contribute?
count(data_above_30, USERID)
# User 6:  430 bad readings out of 976 total observations → device likely buggy


#### Remove User 6 (faulty device) ####
combinedSets <- combinedSets %>% filter(USERID != 6)


#### Cap all signals above -30 to -100 (treat as noise) ####
n_waps <- ncol(combinedSets) - 10
combinedSets_waps  <- as_tibble(apply(combinedSets[, 1:n_waps], c(1, 2), function(x) ifelse(x > -30, -100, x)))
combinedSets_other <- combinedSets[, (n_waps + 1):ncol(combinedSets)]
combinedSets       <- cbind(combinedSets_waps, combinedSets_other)


#### Re-remove zero-variance cols after User 6 deletion ####
combinedSets    <- remove_no_signal_col(combinedSets)
combinedSets_byrow <- remove_no_signal_row(combinedSets)



#+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=
#### NORMALIZATION ####
#+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=

n_waps <- ncol(combinedSets) - 10

combinedSets_wap_all   <- combinedSets[, 1:n_waps]
combinedSets_wap_byrow <- combinedSets_byrow[, 1:(ncol(combinedSets_byrow) - 10)]
combinedSets_wap_bycol <- combinedSets[, 1:n_waps]
combinedSets_other     <- combinedSets[, (n_waps + 1):ncol(combinedSets)]


# Normalize across all WAP values
normalize_all <- function(x) {
  (x - min(combinedSets_wap_all)) / (max(combinedSets_wap_all) - min(combinedSets_wap_all))
}

# Normalize within a single row or column
normalize <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

combinedSets_wap_normalized_all   <- as_tibble(apply(combinedSets_wap_all, 2, normalize_all))
combinedSets_wap_normalized_byrow <- as_tibble(t(apply(combinedSets_wap_byrow, 1, normalize)))
combinedSets_wap_normalized_bycol <- as_tibble(apply(combinedSets_wap_bycol, 2, normalize))

# Sanity check for NAs
sum(!complete.cases(combinedSets_wap_normalized_all))
sum(!complete.cases(combinedSets_wap_normalized_byrow))
sum(!complete.cases(combinedSets_wap_normalized_bycol))

# Align row-normalised metadata with the reduced row set
index                       <- rownames(combinedSets_wap_byrow)
combinedSets_other_by_row   <- combinedSets_other[index, ]

# Assemble full normalised datasets
combinedSets_normalized_all   <- cbind(combinedSets_wap_normalized_all, combinedSets_other)
combinedSets_normalized_byrow <- cbind(combinedSets_wap_normalized_byrow, combinedSets_other_by_row)
combinedSets_normalized_bycol <- cbind(combinedSets_wap_normalized_bycol, combinedSets_other)



#+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=
#### TRAIN / VALIDATION SPLIT ####
#+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=

# Row-wise normalisation produced the best model performance
training   <- combinedSets_normalized_byrow %>% filter(PARTITION == "train")
validation <- combinedSets_normalized_byrow %>% filter(PARTITION == "val")

# Sample 3000 records from the training set (SVM is expensive on 520-dim data)
set.seed(123)
data_index_sampled  <- sample(1:nrow(training), 3000)
training_sampled    <- training[data_index_sampled, ]

# Quick sanity plots
ggplot(training_sampled, aes(x = BUILDINGID, y = FLOOR)) +
  geom_jitter(width = 0.2, alpha = 0.4) +
  labs(title = "Floor Distribution per Building") + theme_minimal()

ggplot(training_sampled, aes(x = BUILDINGID, y = LONGITUDE)) +
  geom_jitter(width = 0.2, alpha = 0.4) +
  labs(title = "Longitude Distribution per Building") + theme_minimal()



#+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=
#### BUILDING ID — CLASSIFICATION ####
#+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=

set.seed(123)
ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

inTrain_building <- createDataPartition(y = training_sampled$BUILDINGID, p = 0.75, list = FALSE)
building_index   <- grep("BUILDINGID", colnames(training_sampled))

training_part_building <- training_sampled[ inTrain_building, ][, c(1:(ncol(training_sampled) - 10), building_index)]
testing_part_building  <- training_sampled[-inTrain_building, ][, c(1:(ncol(training_sampled) - 10), building_index)]


#### KNN ####
training_knn_building <- train(
  BUILDINGID ~ .,
  data      = training_part_building,
  method    = "knn",
  preProc   = c("center", "scale"),
  tuneLength = 10,
  trControl = ctrl,
  metric    = "Accuracy"
)
# k=5: Accuracy=0.984, Kappa=0.975 (train) | Accuracy=0.979, Kappa=0.967 (test)
knn_pred_building <- predict(training_knn_building, testing_part_building)
postResample(knn_pred_building, testing_part_building$BUILDINGID)


#### SVM (Linear) ####
training_svm_building <- train(
  BUILDINGID ~ .,
  data      = training_part_building,
  method    = "svmLinear",
  preProc   = c("center", "scale"),
  tuneLength = 10,
  trControl = ctrl,
  metric    = "Accuracy"
)
# Accuracy=1.000, Kappa=1.000 (both train and test)
svm_pred_building <- predict(training_svm_building, testing_part_building)
postResample(svm_pred_building, testing_part_building$BUILDINGID)


#### SVM3 (Linear with L1/L2) ####
training_svm3_building <- train(
  BUILDINGID ~ .,
  data      = training_part_building,
  method    = "svmLinear3",
  preProc   = c("center", "scale"),
  tuneLength = 10,
  trControl = ctrl,
  metric    = "Accuracy"
)
# cost=0.25, L1: Accuracy=1.000, Kappa=1.000
svm3_pred_building <- predict(training_svm3_building, testing_part_building)
postResample(svm3_pred_building, testing_part_building$BUILDINGID)


#### C5.0 ####
system.time(
  training_c5.0_building <- train(
    BUILDINGID ~ .,
    data      = training_part_building,
    method    = "C5.0",
    preProc   = c("center", "scale"),
    tuneLength = 10,
    trControl = ctrl,
    metric    = "Kappa"
  )
)
# rules, winnow=TRUE, trials=20: Accuracy=0.9996, Kappa=0.9993
c5.0_pred_building <- predict(training_c5.0_building, testing_part_building)
postResample(c5.0_pred_building, testing_part_building$BUILDINGID)


#### Validate Building (SVM3 selected — best performer) ####
building_validation      <- validation[, which(names(validation) %in% names(testing_part_building))]
svm3_validation_building <- predict(training_svm3_building, building_validation)
postResample(svm3_validation_building, building_validation$BUILDINGID)
# Accuracy=1.000, Kappa=1.000



#+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=
#### FLOOR — CLASSIFICATION (Whole Campus) ####
#+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=

inTrain_floor <- createDataPartition(y = training_sampled$FLOOR, p = 0.75, list = FALSE)
floor_index   <- grep("FLOOR", colnames(training_sampled))

training_part_floor <- training_sampled[ inTrain_floor, ][, c(1:(ncol(training_sampled) - 10), floor_index)]
testing_part_floor  <- training_sampled[-inTrain_floor, ][, c(1:(ncol(training_sampled) - 10), floor_index)]


#### SVM (selected — best floor accuracy) ####
system.time(
  training_svm_floor <- train(
    FLOOR ~ .,
    data      = training_part_floor,
    method    = "svmLinear",
    preProc   = c("center", "scale"),
    tuneLength = 10,
    trControl = ctrl,
    metric    = "Accuracy"
  )
)
# Accuracy=0.985, Kappa=0.980 (train) | Accuracy=0.987, Kappa=0.982 (test)
svm_pred_floor <- predict(training_svm_floor, testing_part_floor)
postResample(svm_pred_floor, testing_part_floor$FLOOR)


#### Validate Floor ####
floor_validation            <- validation
floor_validation$BUILDINGID <- svm3_validation_building   # Use predicted Building IDs
floor_validation            <- floor_validation[, which(names(floor_validation) %in% names(testing_part_floor))]
svm_validation_floor        <- predict(training_svm_floor, floor_validation)
postResample(svm_validation_floor, floor_validation$FLOOR)
# Accuracy=0.882, Kappa=0.836



#+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=
#### FLOOR — PER BUILDING ####
#+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=

train_floor_per_building <- function(building_id, training_sampled, validation,
                                     svm3_validation_building, floor_index, ctrl) {
  # Subset training
  train_b <- training_sampled %>% filter(BUILDINGID == building_id)
  train_b$FLOOR <- factor(train_b$FLOOR)  # Drop unused levels

  train_part <- train_b[, c(1:(ncol(train_b) - 10), floor_index)]

  model <- train(
    FLOOR ~ .,
    data      = train_part,
    method    = "svmLinear",
    preProc   = c("center", "scale"),
    tuneLength = 10,
    trControl = ctrl,
    metric    = "Accuracy"
  )

  # Validation
  val_b            <- validation
  val_b$BUILDINGID <- svm3_validation_building
  val_b            <- val_b %>% filter(BUILDINGID == building_id)
  val_b            <- val_b[, which(names(val_b) %in% names(train_part))]

  preds  <- predict(model, val_b)
  result <- postResample(preds, val_b$FLOOR)

  list(model = model, validation_metrics = result)
}

floor_B0 <- train_floor_per_building(0, training_sampled, validation, svm3_validation_building, floor_index, ctrl)
floor_B0$validation_metrics  # Accuracy=0.944, Kappa=0.921

floor_B1 <- train_floor_per_building(1, training_sampled, validation, svm3_validation_building, floor_index, ctrl)
floor_B1$validation_metrics  # Accuracy=0.797, Kappa=0.712

floor_B2 <- train_floor_per_building(2, training_sampled, validation, svm3_validation_building, floor_index, ctrl)
floor_B2$validation_metrics  # Accuracy=0.866, Kappa=0.817



#+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=
#### LATITUDE — REGRESSION ####
#+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=

inTrain_latitude <- createDataPartition(y = training_sampled$LATITUDE, p = 0.75, list = FALSE)
latitude_index   <- grep("LATITUDE", colnames(training_sampled))

training_part_latitude <- training_sampled[ inTrain_latitude, ][, c(1:(ncol(training_sampled) - 10), latitude_index)]
testing_part_latitude  <- training_sampled[-inTrain_latitude, ][, c(1:(ncol(training_sampled) - 10), latitude_index)]


#### KNN ####
training_knn_latitude <- train(
  LATITUDE ~ .,
  data      = training_part_latitude,
  method    = "knn",
  preProc   = c("center", "scale"),
  tuneLength = 10,
  trControl = ctrl,
  metric    = "RMSE"
)
# k=5: RMSE=6.85, R²=0.989, MAE=4.44 (train) | RMSE=7.07, R²=0.988, MAE=4.52 (test)
knn_pred_latitude <- predict(training_knn_latitude, testing_part_latitude)
postResample(knn_pred_latitude, testing_part_latitude$LATITUDE)


#### SVM (Linear) ####
system.time(
  training_svm_latitude <- train(
    LATITUDE ~ .,
    data      = training_part_latitude,
    method    = "svmLinear",
    preProc   = c("center", "scale"),
    tuneLength = 10,
    trControl = ctrl,
    metric    = "RMSE"
  )
)
# RMSE=20.65, R²=0.908, MAE=13.73 (train) | RMSE=18.87, R²=0.919, MAE=11.28 (test)
svm_pred_latitude <- predict(training_svm_latitude, testing_part_latitude)
postResample(svm_pred_latitude, testing_part_latitude$LATITUDE)


#### Validate Latitude (KNN selected — best RMSE) ####
latitude_validation        <- validation[, which(names(validation) %in% names(testing_part_latitude))]
knn_validation_latitude    <- predict(training_knn_latitude, latitude_validation)
postResample(knn_validation_latitude, latitude_validation$LATITUDE)
# RMSE=12.56, R²=0.968, MAE=7.49



#+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=
#### LONGITUDE — REGRESSION ####
#+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=

inTrain_longitude <- createDataPartition(y = training_sampled$LONGITUDE, p = 0.75, list = FALSE)
longitude_index   <- grep("LONGITUDE", colnames(training_sampled))

training_part_longitude <- training_sampled[ inTrain_longitude, ][, c(1:(ncol(training_sampled) - 10), longitude_index)]
testing_part_longitude  <- training_sampled[-inTrain_longitude, ][, c(1:(ncol(training_sampled) - 10), longitude_index)]


#### KNN ####
training_knn_longitude <- train(
  LONGITUDE ~ .,
  data      = training_part_longitude,
  method    = "knn",
  preProc   = c("center", "scale"),
  tuneLength = 10,
  trControl = ctrl,
  metric    = "RMSE"
)
# k=5: RMSE=9.13, R²=0.995, MAE=5.30 (train) | RMSE=6.65, R²=0.997, MAE=4.47 (test)
knn_pred_longitude <- predict(training_knn_longitude, testing_part_longitude)
postResample(knn_pred_longitude, testing_part_longitude$LONGITUDE)


#### SVM (Linear) ####
system.time(
  training_svm_longitude <- train(
    LONGITUDE ~ .,
    data      = training_part_longitude,
    method    = "svmLinear",
    preProc   = c("center", "scale"),
    tuneLength = 10,
    trControl = ctrl,
    metric    = "RMSE"
  )
)
# RMSE=34.07, R²=0.928, MAE=22.79 (train) | RMSE=32.20, R²=0.936, MAE=22.12 (test)
svm_pred_longitude <- predict(training_svm_longitude, testing_part_longitude)
postResample(svm_pred_longitude, testing_part_longitude$LONGITUDE)


#### Validate Longitude (KNN selected) ####
longitude_validation     <- validation[, which(names(validation) %in% names(testing_part_longitude))]
knn_validation_longitude <- predict(training_knn_longitude, longitude_validation)
postResample(knn_validation_longitude, longitude_validation$LONGITUDE)
# RMSE=19.31, R²=0.975, MAE=8.02



#+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=
#### ERROR ANALYSIS — EUCLIDEAN DISTANCE (Lat/Long) ####
#+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=

# Euclidean distance between predicted and actual coordinates (in metres)
errors_lat  <- as.numeric(latitude_validation$LATITUDE)  - as.numeric(knn_validation_latitude)
errors_long <- as.numeric(longitude_validation$LONGITUDE) - as.numeric(knn_validation_longitude)
euclidean   <- sqrt(errors_lat^2 + errors_long^2)

cat("Mean error (m):  ", round(mean(euclidean), 2), "\n")
cat("Median error (m):", round(median(euclidean), 2), "\n")

# Plot actual vs predicted coordinates
actual_df    <- data.frame(LONGITUDE = as.numeric(latitude_validation$LATITUDE),
                           LATITUDE  = as.numeric(longitude_validation$LONGITUDE), type = "Actual")
predicted_df <- data.frame(LONGITUDE = as.numeric(knn_validation_latitude),
                           LATITUDE  = as.numeric(knn_validation_longitude), type = "Predicted")
coords_df    <- rbind(actual_df, predicted_df)

ggplot(coords_df, aes(x = LONGITUDE, y = LATITUDE, colour = type)) +
  geom_point(alpha = 0.4, size = 1) +
  scale_colour_manual(values = c(Actual = "#2166ac", Predicted = "#d6604d")) +
  labs(title = "Actual vs Predicted Coordinates", colour = NULL) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))
