library(keras)

imdb <- dataset_imdb(num_words = 10000)
c(c(train_data, train_labels), c(test_data, test_labels)) %<-% imdb

# word_index <- dataset_imdb_word_index()
# reverse_word_index <- names(word_index)
# names(reverse_word_index) <- word_index
# reverse_word_index[as.character(41)]

vectorize_sequences <- function(sequences, dimension = 10000) {
    results <- matrix(0, nrow = length(sequences), ncol = dimension)
    for (i in 1:length(sequences))
        results[i, sequences[[i]]] <- 1
    results
}

x_train <- vectorize_sequences(train_data)
x_test <- vectorize_sequences(test_data)

y_train <- as.numeric(train_labels)
y_test <- as.numeric(test_labels)

model <- keras_model_sequential() %>%
    layer_dense(units = 8, activation = "relu", input_shape = c(10000)) %>%
#    layer_dense(units = 32, activation = "relu") %>%
#    layer_dense(units = 16, activation = "relu") %>%
    layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
    optimizer = "rmsprop",
    loss = "binary_crossentropy",
    metrics = c("accuracy"))

val_indices <- 1:10000

x_val <- x_train[val_indices,]
partial_x_train <- x_train[-val_indices,]
y_val <- y_train[val_indices]
partial_y_train <- y_train[-val_indices]

history <- model %>% fit(
    partial_x_train,
    partial_y_train,
    epochs = 20,
    batch_size = 512,
    validation_data = list(x_val, y_val)
)

model2 <- keras_model_sequential() %>%
    layer_dense(units = 16, activation = "relu", input_shape = c(10000)) %>%
    layer_dense(units = 16, activation = "relu") %>%
    layer_dense(units = 1, activation = "sigmoid")

model2 %>% compile(
    optimizer = "rmsprop",
    loss = "binary_crossentropy",
    metrics = c("accuracy"))

history <- model2 %>% fit(
    partial_x_train,
    partial_y_train,
    epochs = 4,
    batch_size = 512,
    validation_data = list(x_val, y_val)
)

model %>% predict(x_test[1:10,])

# Wrapping up
# - You usually need to do quite a bit of preprocessing on your raw data in order 
# to be able to feed it—as tensors—into a neural network. Sequences of words can 
# be encoded as binary vectors, but there are other encoding options, too.
# - Stacks of dense layers with relu activations can solve a wide range of problems
# (including sentiment classification), and you’ll likely use them frequently.
# - In a binary classification problem (two output classes), your network should 
# end with a dense layer with one unit and a sigmoid activation: the output of 
# your network should be a scalar between 0 and 1, encoding a probability.
# - With such a scalar sigmoid output on a binary classification problem, the loss
# function you should use is binary_crossentropy.
# - The rmsprop optimizer is generally a good enough choice, whatever your problem.
# That’s one less thing for you to worry about.
# - As they get better on their training data, neural networks eventually start 
# overfitting and end up obtaining increasingly worse results on data they’ve never
# seen before. Be sure to always monitor performance on data that is outside of 
# the training set.
