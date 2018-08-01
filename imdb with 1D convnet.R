library(keras)

max_features <- 10000
max_len <- 500

cat("Loading data...\n")
imdb <- dataset_imdb(num_words = max_features)
c(c(x_train, y_train), c(x_test, y_test)) %<-% imdb

cat(length(x_train), "train sequences\n")
cat(length(x_test), "test sequences")

cat("Pad sequences (samples x time)\n")
x_train <- pad_sequences(x_train, maxlen = max_len)
x_test <- pad_sequences(x_test, maxlen = max_len)

cat("x_train shape:", dim(x_train), "\n")
cat("x_test shape:", dim(x_test), "\n")

model <- keras_model_sequential() %>%
    layer_embedding(input_dim = max_features, output_dim = 128,
                    input_length = max_len) %>%
    layer_conv_1d(filters = 32, kernel_size = 7, activation = "relu") %>%
    layer_max_pooling_1d(pool_size = 5) %>%
    layer_conv_1d(filters = 32, kernel_size = 7, activation = "relu") %>%
    layer_global_max_pooling_1d() %>%
    layer_dense(units = 1)

summary(model)

model %>% compile(
    optimizer = optimizer_rmsprop(lr = 1e-4),
    loss = "binary_crossentropy",
    metrics = c("acc")
)

history <- model %>% fit(
    x_train, y_train,
    epochs = 10,
    batch_size = 128,
    validation_split = 0.2
)

# val_loss: 0.4143 - val_acc: 0.8346 - best result for epoch 6