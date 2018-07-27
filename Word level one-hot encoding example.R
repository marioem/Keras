# Non-keras solution
#
samples <- c("The cat sat on the mat.", "The dog ate my homework.")
token_index <- list()

for (sample in samples)
    for (word in strsplit(sample, " ")[[1]])
        if (!word %in% names(token_index))
            token_index[[word]] <- length(token_index) + 2

max_length <- 10
# Prepare an empty results array for one-hot encoding
results <- array(0, dim = c(length(samples),
                            max_length,
                            max(as.integer(token_index))))

for (i in 1:length(samples)) {
    sample <- samples[[i]]
    words <- head(strsplit(sample, " ")[[1]], n = max_length)
    for (j in 1:length(words)) {
        index <- token_index[[words[[j]]]]
        results[[i, j, index]] <- 1
    }
}

# Keras-solution
#
library(keras)

samples <- c("The cat sat on the mat.", "The dog ate my homework.")
tokenizer <- text_tokenizer(num_words = 1000) %>%
fit_text_tokenizer(samples)

sequences <- texts_to_sequences(tokenizer, samples)

one_hot_results <- texts_to_matrix(tokenizer, samples, mode = "binary")
dim(one_hot_results)
word_index <- tokenizer$word_index
cat("Found", length(word_index), "unique tokens.\n")

# Adding a new sample
# After tokenizer is updated the tokens indexes may change so all the samples need
# to be recoded.
new_sample <- "Ala ma kota"
tokenizer <- tokenizer %>% fit_text_tokenizer(new_sample)
sequences <- texts_to_sequences(tokenizer, new_sample)

one_hot_results <- texts_to_matrix(tokenizer, c(samples, new_sample), mode = "binary")
dim(one_hot_results)

word_index <- tokenizer$word_index
cat("Found", length(word_index), "unique tokens.\n")
