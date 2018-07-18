library(keras)

mnist <- dataset_mnist()
x_train <- mnist$train$x       # x - Images
y_train <- mnist$train$y       # y - Labels
x_test <- mnist$test$x
y_test <- mnist$test$y

# reshape
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))
# rescale
x_train <- x_train / 255
x_test <- x_test / 255

# Let's peek into the training set

for(i in 1:10){
    digit <- mnist$train$x[i,,]
    digit <- 255 - digit
    if(i == 1)
        digits <- digit
    else
        digits <- cbind(digits, digit)
}
plot(as.raster(digits, max = 255))

y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

model <- keras_model_sequential() 
model %>% 
    layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>% 
    layer_dropout(rate = 0.4) %>% 
    layer_dense(units = 128, activation = 'relu') %>%
    layer_dropout(rate = 0.3) %>%
    layer_dense(units = 10, activation = 'softmax')

model %>% compile(
    loss = 'categorical_crossentropy',
    optimizer = optimizer_rmsprop(),
    metrics = c('accuracy')
)

history <- model %>% fit(
    x_train, y_train, 
    epochs = 30, batch_size = 128, # The bigger the batch the better accuracy 
    validation_split = 0.2
)

plot(history)

model %>% evaluate(x_test, y_test)

classes <- model %>% predict_classes(x_test)

mean(classes == mnist$test$y)

y <- array(5, dim = c(32, 10))

