library(keras)
library(corrplot)

# Read in MNIST data
mnist <- dataset_mnist()

# Read in CIFAR10 data
cifar10 <- dataset_cifar10()

# Read in IMDB data
imdb <- dataset_imdb()

# Store the overall correlation in `M`
M <- cor(iris[,1:4])

# Plot the correlation plot with `M`
corrplot(M, method="circle")
data("iris")
summary(iris)

# Build your own `normalize()` function
normalize <- function(x) {
    num <- x - min(x)
    denom <- max(x) - min(x)
    return (num/denom)
}

# Normalize the `iris` data
iris_norm <- as.data.frame(lapply(iris[1:4], normalize))

# Return the first part of `iris` 
head(iris_norm)
summary(iris_norm)

iris_m <- iris
iris_m[,5] <- as.numeric(iris_m[,5]) -1

# Turn `iris` into a matrix
iris_m <- as.matrix(iris_m)

# Set `iris` `dimnames` to `NULL`
dimnames(iris_m) <- NULL

# Normalize the `iris` data
iris_m[,1:4] <- normalize(iris_m[,1:4])

# Return the summary of `iris`
summary(iris_m)

# Determine sample size
ind <- sample(2, nrow(iris_m), replace=TRUE, prob=c(0.67, 0.33))

# Split the `iris` data
iris.training <- iris_m[ind==1, 1:4]
iris.test <- iris_m[ind==2, 1:4]

# Split the class attribute
iris.trainingtarget <- iris_m[ind==1, 5]
iris.testtarget <- iris_m[ind==2, 5]

# One hot encode training target values
iris.trainLabels <- to_categorical(iris.trainingtarget)

# One hot encode test target values
iris.testLabels <- to_categorical(iris.testtarget)

# Print out the iris.testLabels to double check the result
print(iris.testLabels)

# Initialize a sequential model
model <- keras_model_sequential()

# Add layers to the model
model %>% 
    layer_dense(units = 8, activation = 'relu', input_shape = c(4)) %>% 
    layer_dense(units = 3, activation = 'softmax')

# Print a summary of a model
summary(model)

# Get model configuration
get_config(model)

# Get layer configuration
get_layer(model, index = 1)

# List the model's layers
model$layers

# List the input tensors
model$inputs

# List the output tensors
model$outputs

# Compile the model
model %>% compile(
    loss = 'categorical_crossentropy',
    optimizer = 'adam',
    metrics = 'accuracy'
)

# Fit the model 
model %>% fit(
    iris.training, 
    iris.trainLabels, 
    epochs = 200, 
    batch_size = 5, 
    validation_split = 0.2
)

# Store the fitting history in `history` 
history <- model %>% fit(
    iris.training, 
    iris.trainLabels, 
    epochs = 200,
    batch_size = 5, 
    validation_split = 0.2
)

# Plot the history
plot(history)

# Plot the model loss of the test data. Usually this loss is bigger so we need to print it first so that the plot gets the correct ranges
plot(history$metrics$val_loss, main="Model Loss", xlab = "epoch", ylab="loss", col="green", type="l", ylim = c(0, max(history$metrics$val_loss)))

# Plot the model loss of the treaining data
lines(history$metrics$loss, col="blue")

# Add legend
legend("topright", c("train","test"), col=c("blue", "green"), lty=c(1,1))

# Plot the accuracy of the training data 
plot(history$metrics$acc, main="Model Accuracy", xlab = "epoch", ylab="accuracy", col="blue", type="l", ylim = c(0, max(history$metrics$acc)))

# Plot the accuracy of the validation data
lines(history$metrics$val_acc, col="green")

# Add Legend
legend("bottomright", c("train","test"), col=c("blue", "green"), lty=c(1,1))

# Predict the classes for the test data
classes <- model %>% predict_classes(iris.test, batch_size = 128)

# Confusion matrix
table(iris.testtarget, classes)

# Evaluate on test data and labels
score <- model %>% evaluate(iris.test, iris.testLabels, batch_size = 128)

# Print the score
print(score)

################################################################################
# Model tweaking
################################################################################

################################################################################
# Adding layers
################################################################################

# Initialize the sequential model
model <- keras_model_sequential() 

# Add layers to model
model %>% 
    layer_dense(units = 8, activation = 'relu', input_shape = c(4)) %>% 
    layer_dense(units = 5, activation = 'relu') %>% 
    layer_dense(units = 3, activation = 'softmax')

# Compile the model
model %>% compile(
    loss = 'categorical_crossentropy',
    optimizer = 'adam',
    metrics = 'accuracy'
)

# Fit the model to the data
model %>% fit(
    iris.training, iris.trainLabels, 
    epochs = 200, batch_size = 5, 
    validation_split = 0.2
)

# Evaluate the model
score <- model %>% evaluate(iris.test, iris.testLabels, batch_size = 128)

# Print the score
print(score)

# Save the training history in history
history <- model %>% fit(
    iris.training, iris.trainLabels, 
    epochs = 200, batch_size = 5,
    validation_split = 0.2
)

# Plot the model loss
plot(history$metrics$loss, main="Model Loss", xlab = "epoch", ylab="loss", col="blue", type="l", ylim=c(0, 1))
lines(history$metrics$val_loss, col="green")
legend("topright", c("train","test"), col=c("blue", "green"), lty=c(1,1))

# Plot the model accuracy
plot(history$metrics$acc, main="Model Accuracy", xlab = "epoch", ylab="accuracy", col="blue", type="l", ylim = c(0, 1))
lines(history$metrics$val_acc, col="green")
legend("bottomright", c("train","test"), col=c("blue", "green"), lty=c(1,1))

################################################################################
# Adding hidden units
################################################################################

# Initialize a sequential model
model <- keras_model_sequential() 

# Add layers to the model
model %>% 
    layer_dense(units = 28, activation = 'relu', input_shape = c(4)) %>% 
    layer_dense(units = 3, activation = 'softmax')

# Compile the model
model %>% compile(
    loss = 'categorical_crossentropy',
    optimizer = 'adam',
    metrics = 'accuracy'
)

# Fit the model to the data
model %>% fit(
    iris.training, iris.trainLabels, 
    epochs = 200, batch_size = 5, 
    validation_split = 0.2
)

# Evaluate the model
score <- model %>% evaluate(iris.test, iris.testLabels, batch_size = 128)

# Print the score
print(score)

# Save the training history in the history variable
history <- model %>% fit(
    iris.training, iris.trainLabels, 
    epochs = 200, batch_size = 5, 
    validation_split = 0.2
)

# Plot the model loss
plot(history$metrics$loss, main="Model Loss", xlab = "epoch", ylab="loss", col="blue", type="l", ylim = c(0,1))
lines(history$metrics$val_loss, col="green")
legend("topright", c("train","test"), col=c("blue", "green"), lty=c(1,1))

# Plot the model accuracy
plot(history$metrics$acc, main="Model Accuracy", xlab = "epoch", ylab="accuracy", col="blue", type="l", ylim = c(0,1))
lines(history$metrics$val_acc, col="green")
legend("bottomright", c("train","test"), col=c("blue", "green"), lty=c(1,1))

################################################################################
# Trying different optimizer
################################################################################

# Initialize a sequential model
model <- keras_model_sequential() 

# Build up your model by adding layers to it
model %>% 
    layer_dense(units = 8, activation = 'relu', input_shape = c(4)) %>% 
    layer_dense(units = 3, activation = 'softmax')

# Define an optimizer
sgd <- optimizer_sgd(lr = 0.01)

# Use the optimizer to compile the model
model %>% compile(optimizer=sgd, 
                  loss='categorical_crossentropy', 
                  metrics='accuracy')

# Fit the model to the training data
history <- model %>% fit(
    iris.training, iris.trainLabels, 
    epochs = 200, batch_size = 5, 
    validation_split = 0.2
)

# Evaluate the model
score <- model %>% evaluate(iris.test, iris.testLabels, batch_size = 128)

# Print the loss and accuracy metrics
print(score)

# Plot the model loss
plot(history$metrics$loss, main="Model Loss", xlab = "epoch", ylab="loss", col="blue", type="l", ylim = c(0,1))
lines(history$metrics$val_loss, col="green")
legend("topright", c("train","test"), col=c("blue", "green"), lty=c(1,1))

# Plot the model accuracy
plot(history$metrics$acc, main="Model Accuracy", xlab = "epoch", ylab="accuracy", col="blue", type="l", ylim = c(0,1))
lines(history$metrics$val_acc, col="green")
legend("bottomright", c("train","test"), col=c("blue", "green"), lty=c(1,1))

summary(model)

# save_model_hdf5(model, "my_model.h5")
# model <- load_model_hdf5("my_model.h5")

# save_model_weights_hdf5("my_model_weights.h5")
# model %>% load_model_weights_hdf5("my_model_weights.h5")

# json_string <- model_to_json(model)
# model <- model_from_json(json_string)
# 
# yaml_string <- model_to_yaml(model)
# model <- model_from_yaml(yaml_string)

