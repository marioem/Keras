library(keras)

dataset <- dataset_boston_housing()
c(c(train_data, train_targets), c(test_data, test_targets)) %<-% dataset

# Feature normalization, before feeding it in into ANN

mean <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)
train_data <- scale(train_data, center = mean, scale = std)
test_data <- scale(test_data, center = mean, scale = std) # scales test data using train stats

build_model <- function() {                                1
    model <- keras_model_sequential() %>%
        layer_dense(units = 64, activation = "relu",
                    input_shape = dim(train_data)[[2]]) %>%
        layer_dense(units = 64, activation = "relu") %>%
        layer_dense(units = 1)
    model %>% compile(
        optimizer = "rmsprop",
        loss = "mse",
        metrics = c("mae")
    )
}

# Setting up k-fold validation

k <- 4
indices <- sample(1:nrow(train_data))
folds <- cut(indices, breaks = k, labels = FALSE)

num_epochs <- 500
all_mae_histories <- NULL
for (i in 1:k) {
    cat("processing fold #", i, "\n")
    
    val_indices <- which(folds == i, arr.ind = TRUE)
    val_data <- train_data[val_indices,]
    val_targets <- train_targets[val_indices]
    
    partial_train_data <- train_data[-val_indices,]
    partial_train_targets <- train_targets[-val_indices]
    
    model <- build_model()
    
    history <- model %>% fit(partial_train_data, partial_train_targets,
                             validation_data = list(val_data, val_targets),
                             epochs = num_epochs, batch_size = 1, verbose = 0)
    
    mae_history <- history$metrics$val_mean_absolute_error
    all_mae_histories <- rbind(all_mae_histories, mae_history)
}

average_mae_history <- data.frame(
    epoch = seq(1:ncol(all_mae_histories)),
    validation_mae = apply(all_mae_histories, 2, mean)
)

library(ggplot2)
ggplot(average_mae_history, aes(x = epoch, y = validation_mae)) + geom_line()

ggplot(average_mae_history, aes(x = epoch, y = validation_mae)) + geom_smooth()

# Training the final model
model <- build_model()
model %>% fit(train_data, train_targets,
              epochs = 115, batch_size = 16, verbose = 0)
result <- model %>% evaluate(test_data, test_targets)

result

# Wrapping up
# - Regression is done using different loss functions than classification. Mean 
# squared error (MSE) is a loss function commonly used for regression.
# - Similarly, evaluation metrics to be used for regression differ from those used 
# for classification; naturally, the concept of accuracy doesn’t apply for 
# regression. A common regression metric is mean absolute error (MAE).
# - When features in the input data have values in different ranges, each feature
# should be scaled independently as a preprocessing step.
# - When there is little data available, using K-fold validation is a great way to
# reliably evaluate a model.
# - When little training data is available, it’s preferable to use a small network
# with few hidden layers (typically only one or two), in order to avoid severe
# overfitting.
