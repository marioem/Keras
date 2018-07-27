#
# What is happening here.
# We're reading in the original IMDB reviews, tokenizing them and assigning them
# labels. Then the samples (review, label) are split into training and validation
# sets.
# Next GloVe word embeddings are read - it consists of 400000 words with their 
# 100-dimensional embeddings vectors. It seems that those embeddings can be used
# as the embeddings layer weights, so the next step is to create a matrix of those
# embeddings vectors, that matrix is then used to initialize the weights of the
# embedding layer.
# Then the model predicting sentiment of a review is trained on only 200 samples
# on that pretrained embeddings network.
# Question. Why pretrained embedding layer should in any way help in achieving
# better prediction accuracy for this problem???
#
library(keras)

imdb_dir <- "~/Documents/GitHub/Keras/aclImdb"
train_dir <- file.path(imdb_dir, "train")
labels <- c()
texts <- c()
for (label_type in c("neg", "pos")) {
    label <- switch(label_type, neg = 0, pos = 1)
    dir_name <- file.path(train_dir, label_type)
    for (fname in list.files(dir_name, pattern = glob2rx("*.txt"),
                             full.names = TRUE)) {
        texts <- c(texts, readChar(fname, file.info(fname)$size))
        labels <- c(labels, label)
    }
}

# Tokenizing the text

maxlen <- 100
training_samples <- 200
validation_samples <- 10000
max_words <- 10000

tokenizer <- text_tokenizer(num_words = max_words) %>%
    fit_text_tokenizer(texts)
sequences <- texts_to_sequences(tokenizer, texts)
word_index = tokenizer$word_index

cat("Found", length(word_index), "unique tokens.\n")
data <- pad_sequences(sequences, maxlen = maxlen)

labels <- as.array(labels)
cat("Shape of data tensor:", dim(data), "\n")
cat('Shape of label tensor:', dim(labels), "\n")
indices <- sample(1:nrow(data))
training_indices <- indices[1:training_samples]
validation_indices <- indices[(training_samples + 1):
                                  (training_samples + validation_samples)]
x_train <- data[training_indices,]
y_train <- labels[training_indices]
x_val <- data[validation_indices,]
y_val <- labels[validation_indices]

# Parsing the GloVe word-embeddings file
#
glove_dir = "~/Documents/GitHub/Keras/GloVe/glove.6B"
lines <- readLines(file.path(glove_dir, "glove.6B.100d.txt"))
embeddings_index <- new.env(hash = TRUE, parent = emptyenv())
for (i in 1:length(lines)) {
    line <- lines[[i]]
    values <- strsplit(line, " ")[[1]]
    word <- values[[1]]
    embeddings_index[[word]] <- as.double(values[-1])
}
cat("Found", length(embeddings_index), "word vectors.\n")

# Preparing the GloVe word-embeddings matrix
#
embedding_dim <- 100
embedding_matrix <- array(0, c(max_words, embedding_dim))

for (word in names(word_index)) {
    index <- word_index[[word]]
    if (index < max_words) {
        embedding_vector <- embeddings_index[[word]]
        if (!is.null(embedding_vector)) # Words not found in the embedding index will be all zeros
            embedding_matrix[index+1,] <- embedding_vector
            # index 1 isn’t supposed to stand for any word or token—it’s a placeholder.
    }
}

model <- keras_model_sequential() %>%
    layer_embedding(input_dim = max_words, output_dim = embedding_dim,
                    input_length = maxlen) %>%
    layer_flatten() %>%
    layer_dense(units = 32, activation = "relu") %>%
    layer_dense(units = 1, activation = "sigmoid")

summary(model)

get_layer(model, index = 0) %>%
    set_weights(list(embedding_matrix)) %>%
    freeze_weights()

model %>% compile(
    optimizer = "rmsprop",
    loss = "binary_crossentropy",
    metrics = c("acc")
)
history <- model %>% fit(
    x_train, y_train,
    epochs = 20,
    batch_size = 32,
    validation_data = list(x_val, y_val)
)
save_model_weights_hdf5(model, "pre_trained_glove_model.h5")

###############################################################################
#
# Model evaluation
#
test_dir <- file.path(imdb_dir, "test")
labels <- c()
texts <- c()

for (label_type in c("neg", "pos")) {
    label <- switch(label_type, neg = 0, pos = 1)
    dir_name <- file.path(test_dir, label_type)
    for (fname in list.files(dir_name, pattern = glob2rx("*.txt"),
                             full.names = TRUE)) {
        texts <- c(texts, readChar(fname, file.info(fname)$size))
        labels <- c(labels, label)
    }
}

sequences <- texts_to_sequences(tokenizer, texts)

x_test <- pad_sequences(sequences, maxlen = maxlen)
y_test <- as.array(labels)

model %>%
    load_model_weights_hdf5("pre_trained_glove_model.h5") %>%
    evaluate(x_test, y_test)

# $loss
# [1] 0.8601566
# 
# $acc
# [1] 0.56116

################################################################################
#
# Trainign the same model without pretrained word embeddings
#
model <- keras_model_sequential() %>%
    layer_embedding(input_dim = max_words, output_dim = embedding_dim,
                    input_length = maxlen) %>%
    layer_flatten() %>%
    layer_dense(units = 32, activation = "relu") %>%
    layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
    optimizer = "rmsprop",
    loss = "binary_crossentropy",metrics = c("acc")
)

history <- model %>% fit(
    x_train, y_train,
    epochs = 20,
    batch_size = 32,
    validation_data = list(x_val, y_val)
)


