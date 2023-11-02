# Load the necessary libraries
library(tidyverse)  # For data manipulation and visualization
library(caret)      # For machine learning and data preprocessing
library(glmnet)     # For lasso and ridge regression
library(themis)     # For SMOTE

# Read the CSV file
data <- read.csv("Data analysis with R/data.csv")

# Handling Missing Values: Remove rows with NA values
data <- na.omit(data)

# Drop the customerID column as it's not relevant for analysis
data <- data %>% select(-customerID)

# Apply one-hot encoding, omitting one category for each categorical variable to avoid multicollinearity
data_encoded <- model.matrix(~ . - 1, data)
data_encoded <- as.data.frame(data_encoded) # Convert the result back to a data frame
data_encoded$genderFemale <- NULL # Remove the 'genderFemale' column

# Split the dataset into a training set and a testing set using an 80-20 split
set.seed(123) # Set a seed for reproducibility
index <- createDataPartition(data_encoded$ChurnYes, p = 0.8, list = FALSE)
train_data <- data_encoded[index, ]
test_data <- data_encoded[-index, ]

# Visualization: Histograms for numerical columns
hist_cols <- c("tenure", "MonthlyCharges", "TotalCharges")
for (col in hist_cols) {
  p <- ggplot(train_data, aes(x = !!sym(col))) + 
    geom_histogram(fill="blue", color="black", alpha=0.7) + 
    labs(title=paste("Histogram of", col), x=col, y="Frequency") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  print(p)
}

# Visualization: Box plots of numerical columns against Churn
for (col in hist_cols) {
  p <- ggplot(train_data, aes(x = as.factor(ChurnYes), y = !!sym(col))) + 
    geom_boxplot(fill="blue", color="black", alpha=0.7) + 
    labs(title=paste("Boxplot of", col, "by Churn"), x="ChurnYes", y=col) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  print(p)
}

# Data Preprocessing: Scale the numerical columns
columns_to_scale <- c("tenure", "MonthlyCharges", "TotalCharges")
preprocess_params <- preProcess(train_data[, columns_to_scale], method = c("range"))
train_data[, columns_to_scale] <- predict(preprocess_params, train_data[, columns_to_scale])
test_data[, columns_to_scale] <- predict(preprocess_params, test_data[, columns_to_scale])

# Feature Selection using Lasso Regression
x_train <- as.matrix(train_data %>% select(-ChurnYes))
y_train <- train_data$ChurnYes
cv_lasso <- cv.glmnet(x_train, y_train, alpha = 1)
lasso_best_lambda <- cv_lasso$lambda.min
lasso_model <- glmnet(x_train, y_train, alpha = 1, lambda = lasso_best_lambda)

# Identify selected features
lasso_coefs <- coef(lasso_model, s = lasso_best_lambda)
selected_features <- rownames(lasso_coefs)[-1][lasso_coefs[-1, 1] != 0] # Exclude intercept

# Prepare Data with Selected Features and Apply SMOTE
train_data_selected <- train_data[selected_features]
train_data_selected$ChurnYes <- as.factor(train_data$ChurnYes)

# Apply SMOTE to the training data
set.seed(123) # Set seed for reproducibility
recipe_obj <- recipe(ChurnYes ~ ., data = train_data_selected) %>%
  step_smote(ChurnYes)
trained_recipe <- prep(recipe_obj)
train_data_smote <- juice(trained_recipe)
train_data_smote$ChurnYes <- factor(train_data_smote$ChurnYes, levels = c("0", "1"), labels = c("No", "Yes"))

# Preparing test data
test_data_selected <- test_data[selected_features]
test_data_selected$ChurnYes <- as.factor(test_data$ChurnYes)

# Train Lasso, Ridge, and Elastic Net models
train_control <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary)
metric <- "ROC"

# Lasso model
lasso_grid <- expand.grid(alpha = 1, lambda = 10^seq(-5, 5, length = 100))
lasso_model_final <- train(ChurnYes ~ ., data = train_data_smote, method = "glmnet", 
                           trControl = train_control, tuneGrid = lasso_grid, metric = metric)

# Ridge model
ridge_grid <- expand.grid(alpha = 0, lambda = 10^seq(-5, 5, length = 100))
ridge_model_final <- train(ChurnYes ~ ., data = train_data_smote, method = "glmnet", trControl = train_control, tuneGrid = ridge_grid, metric = metric)

# Elastic Net model
elastic_grid <- expand.grid(alpha = seq(0, 1, length = 10), lambda = 10^seq(-5, 5, length = 100))
elastic_model_final <- train(ChurnYes ~ ., data = train_data_smote, method = "glmnet", trControl = train_control, tuneGrid = elastic_grid, metric = metric)

# Function to evaluate the model based on different metrics
evaluate_model <- function(model, test_data, threshold) {
  # Predictions
  predictions_prob <- predict(model, newdata=test_data, type="prob")[,2]
  predictions <- ifelse(predictions_prob > threshold, 1, 0)
  
  # Compute metrics from the confusion matrix
  conf_matrix <- table(test_data$ChurnYes, predictions)
  accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
  precision <- conf_matrix[2,2] / (conf_matrix[2,2] + conf_matrix[1,2])
  recall <- conf_matrix[2,2] / (conf_matrix[2,2] + conf_matrix[2,1])
  f1_score <- 2 * ((precision * recall) / (precision + recall))
  roc_obj <- roc(test_data$ChurnYes, predictions_prob)
  auc_value <- auc(roc_obj)
  
  # Display metrics
  cat("Threshold:", threshold, "\n")
  cat("Accuracy:", accuracy, "\n")
  cat("Precision:", precision, "\n")
  cat("Recall:", recall, "\n")
  cat("F1-Score:", f1_score, "\n")
  cat("ROC AUC:", auc_value, "\n\n")
}

# Evaluate each model on the test set
cat("Lasso Model Evaluation:\n")
evaluate_model(lasso_model_final, test_data_selected, 0.5)

cat("Ridge Model Evaluation:\n")
evaluate_model(ridge_model_final, test_data_selected, 0.5)

cat("Elastic Net Model Evaluation:\n")
evaluate_model(elastic_model_final, test_data_selected, 0.5)

# Analyze Data Distribution Before and After SMOTE

# Function to plot the class distribution
plot_distribution <- function(data, title) {
  ggplot(data, aes(x = ChurnYes)) +
    geom_bar(fill = "blue", color = "black") +
    theme_minimal() +
    labs(title = title, x = "Churn", y = "Count") +
    theme(plot.title = element_text(hjust = 0.5))
}

# Plot the distribution in the original training set
p1 <- plot_distribution(train_data, "Class Distribution in Original Training Set")
print(p1)

# Plot the distribution after applying SMOTE
p2 <- plot_distribution(train_data_smote, "Class Distribution After SMOTE")
print(p2)

