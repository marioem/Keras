library(keras)

dir.create("~/Documents/GitHub/Keras/jena_climate", recursive = TRUE)
download.file(
    "https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip",
    "~/Documents/GitHub/Keras/jena_climate/jena_climate_2009_2016.csv.zip"
)
unzip(
    "~/Documents/GitHub/Keras/jena_climate/jena_climate_2009_2016.csv.zip",
    exdir = "~/Documents/GitHub/Keras/jena_climate"
)

library(tibble)
library(readr)

data_dir <- "~/Documents/GitHub/Keras/jena_climate"
fname <- file.path(data_dir, "jena_climate_2009_2016.csv")

data <- read_csv(fname)
glimpse(data)

library(ggplot2)

ggplot(data, aes(x = 1:nrow(data), y = `T (degC)`)) + geom_line()
ggplot(data[1:1440,], aes(x = 1:1440, y = `T (degC)`)) + geom_line()


data <- data.matrix(data[,-1])

# Normalizing the data
#
train_data <- data[1:200000,]
mean <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)
data <- scale(data, center = mean, scale = std)


generator <- function(data, lookback, delay, min_index, max_index,
                       shuffle = FALSE, batch_size = 128, step = 6) {
    if (is.null(max_index)) 
        max_index <- nrow(data) - delay - 1
    i <- min_index + lookback
    function() {
        if (shuffle) {
            rows <- sample(c((min_index+lookback):max_index), size = batch_size)
        } else {
            if (i + batch_size >= max_index)
                i <<- min_index + lookback
            rows <- c(i:min(i+batch_size, max_index))
            i <<- i + length(rows)
        }
        samples <- array(0, dim = c(length(rows),
                                    lookback / step,
                                    dim(data)[[-1]]))
        targets <- array(0, dim = c(length(rows)))
        for (j in 1:length(rows)) {
            indices <- seq(rows[[j]] - lookback, rows[[j]],
                           length.out = dim(samples)[[2]])
            samples[j,,] <- data[indices,]
            targets[[j]] <- data[rows[[j]] + delay,2]
        }
        list(samples, targets)
    }
}

lookback <- 1440
step <- 6
delay <- 144
batch_size <- 128

train_gen <- generator(
    data,
    lookback = lookback,
    delay = delay,
    min_index = 1,
    max_index = 200000,
    shuffle = TRUE,
    step = step,
    batch_size = batch_size
)

val_gen = generator(
    data,
    lookback = lookback,
    delay = delay,
    min_index = 200001,
    max_index = 300000,
    step = step,
    batch_size = batch_size
)

test_gen <- generator(
    data,
    lookback = lookback,
    delay = delay,
    min_index = 300001,
    max_index = NULL,
    step = step,
    batch_size = batch_size
)

val_steps <- (300000 - 200001 - lookback) / batch_size
test_steps <- (nrow(data) - 300001 - lookback) / batch_size


# Naive predictor - baseline

evaluate_naive_method <- function() {
    batch_maes <- c()
    for (step in 1:val_steps) {
        c(samples, targets) %<-% val_gen()
        preds <- samples[,dim(samples)[[2]],2]
        mae <- mean(abs(preds - targets))
        batch_maes <- c(batch_maes, mae)
    }
    print(mean(batch_maes))
}
evaluate_naive_method()

celsius_mae <- 0.29 * std[[2]]
# 2.567231 degree C - average absolute error of naive predictor

################################################################################
#
# Approach 1: Densly connected ANN
#
model <- keras_model_sequential() %>%
    layer_flatten(input_shape = c(lookback / step, dim(data)[-1])) %>%
    layer_dense(units = 32, activation = "relu") %>%
    layer_dense(units = 1)

model %>% compile(
    optimizer = optimizer_rmsprop(),
    loss = "mae"
)

history <- model %>% fit_generator(
    train_gen,
    steps_per_epoch = 500,
    epochs = 20,
    validation_data = val_gen,
    validation_steps = val_steps
)

# val_loss: 0.2889 - best result for epoch 4

################################################################################
#
# Approach 2: GRU
#
model <- keras_model_sequential() %>%
    layer_gru(units = 32, input_shape = list(NULL, dim(data)[[-1]])) %>%
    layer_dense(units = 1)

model %>% compile(
    optimizer = optimizer_rmsprop(),
    loss = "mae"
)

history <- model %>% fit_generator(
    train_gen,
    steps_per_epoch = 500,
    epochs = 20,
    validation_data = val_gen,
    validation_steps = val_steps
)

# val_loss: 0.2636 - best result for epoch 5


################################################################################
#
# Approach 3: GRU with recurrent dropout
#
model <- keras_model_sequential() %>%
    layer_gru(units = 32, dropout = 0.2, recurrent_dropout = 0.2,
               input_shape = list(NULL, dim(data)[[-1]])) %>%
    layer_dense(units = 1)

model %>% compile(
    optimizer = optimizer_rmsprop(),
    loss = "mae"
)

history <- model %>% fit_generator(
    train_gen,
    steps_per_epoch = 500,
    epochs = 40,
    validation_data = val_gen,
    validation_steps = val_steps
)
plot(history)

# val_loss: 0.2546 - best result for epoch 32

################################################################################
#
# Approach 4: stacked GRU with recurrent dropout
#
model <- keras_model_sequential() %>%
    layer_gru(units = 32,
              dropout = 0.1,
              recurrent_dropout = 0.5,
              return_sequences = TRUE,
              input_shape = list(NULL, dim(data)[[-1]])) %>%
    layer_gru(units = 64, activation = "relu",
              dropout = 0.1,
              recurrent_dropout = 0.5) %>%
    layer_dense(units = 1)

model %>% compile(
    optimizer = optimizer_rmsprop(),
    loss = "mae"
)

history <- model %>% fit_generator(
    train_gen,
    steps_per_epoch = 500,
    epochs = 40,
    validation_data = val_gen,
    validation_steps = val_steps
)

plot(history)

# val_loss: 0.2554 - best result for epoch 25

################################################################################
#
# Approach 5: bidirectional GRU
#
model <- keras_model_sequential() %>%
    bidirectional(
        layer_gru(units = 32), input_shape = list(NULL, dim(data)[[-1]])
    ) %>%
    layer_dense(units = 1)

model %>% compile(
    optimizer = optimizer_rmsprop(),
    loss = "mae"
)

history <- model %>% fit_generator(
    train_gen,
    steps_per_epoch = 500,
    epochs = 40,
    validation_data = val_gen,
    validation_steps = val_steps
)

plot(history)

# val_loss: 0.3783 - val_acc: 0.8768 - best result for epoch 4

# Further possible improvements:
# - Adjust the number of units in each recurrent layer in the stacked setup. 
# The current choices are largely arbitrary and thus probably suboptimal.
# - Adjust the learning rate used by the RMSprop optimizer.
# - Try using layer_lstm instead of layer_gru.
# - Try using a bigger densely connected regressor on top of the recurrent layers:
# that is, a bigger dense layer or even a stack of dense layers.
# - Don’t forget to eventually run the best-performing models (in terms of 
# validation MAE) on the test set! Otherwise, you’ll develop architectures that 
# are overfitting to the validation set.
