library(keras)

conv_base <- application_vgg16(
    weights = "imagenet",
    include_top = FALSE,
    input_shape = c(150, 150, 3)
)

conv_base

################################################################################
#
# Approach 1.
# Run trainig data through the pre-trained VGG16 model, save the output then run
# it through a new densly connected classifier model.
#
base_dir <- "~/Documents/GitHub/Keras/Cats and dogs small/"
train_dir <- file.path(base_dir, "train")
validation_dir <- file.path(base_dir, "validation")
test_dir <- file.path(base_dir, "test")
datagen <- image_data_generator(rescale = 1/255)

batch_size <- 20

extract_features <- function(directory, sample_count) {
    features <- array(0, dim = c(sample_count, 4, 4, 512))
    labels <- array(0, dim = c(sample_count))
    generator <- flow_images_from_directory(
        directory = directory,
        generator = datagen,
        target_size = c(150, 150),
        batch_size = batch_size,
        class_mode = "binary"
    )
    i <- 0
    while(TRUE) {
        batch <- generator_next(generator)
        inputs_batch <- batch[[1]]
        labels_batch <- batch[[2]]
        features_batch <- conv_base %>% predict(inputs_batch)
        index_range <- ((i * batch_size)+1):((i + 1) * batch_size)
        features[index_range,,,] <- features_batch
        labels[index_range] <- labels_batch
        i <- i + 1
        if (i * batch_size >= sample_count)
            break
    }
    list(
        features = features,
        labels = labels
    )
}
train <- extract_features(train_dir, 2000)
validation <- extract_features(validation_dir, 1000)
test <- extract_features(test_dir, 1000)

reshape_features <- function(features) {
    array_reshape(features, dim = c(nrow(features), 4 * 4 * 512))
}

train$features <- reshape_features(train$features)
validation$features <- reshape_features(validation$features)
test$features <- reshape_features(test$features)

model <- keras_model_sequential() %>%
    layer_dense(units = 256, activation = "relu",
                input_shape = 4 * 4 * 512) %>%
    layer_dropout(rate = 0.5) %>%
    layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
    optimizer = optimizer_rmsprop(lr = 2e-5),
    loss = "binary_crossentropy",
    metrics = c("accuracy")
)

history <- model %>% fit(
    train$features, train$labels,
    epochs = 30,
    batch_size = 20,
    validation_data = list(validation$features, validation$labels)
)

################################################################################
#
# Approach 2.
# Stack densly connected layers on the conv_base and train the entire model on 
# your examples. Data augmentation might be used in this case.
#

# Need to load conv_base again as Approach 1 touched the model

conv_base <- application_vgg16(
    weights = "imagenet",
    include_top = FALSE,
    input_shape = c(150, 150, 3)
)

model <- keras_model_sequential() %>%
    conv_base %>%
    layer_flatten() %>%
    layer_dense(units = 256, activation = "relu") %>%
    layer_dense(units = 1, activation = "sigmoid")

# Before you compile and train the model, it’s very important to freeze 
# the convolutional base.

cat("This is the number of trainable weights before freezing",
     "the conv base:", length(model$trainable_weights), "\n")
freeze_weights(conv_base)
cat("This is the number of trainable weights after freezing",
      "the conv base:", length(model$trainable_weights), "\n")
    
train_datagen = image_data_generator(
    rescale = 1/255,
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = TRUE,
    fill_mode = "nearest"
)

test_datagen <- image_data_generator(rescale = 1/255)

train_generator <- flow_images_from_directory(train_dir,
                                              train_datagen,
                                              target_size = c(150, 150),
                                              batch_size = 20,
                                              class_mode = "binary"
)

validation_generator <- flow_images_from_directory(
    validation_dir,
    test_datagen,
    target_size = c(150, 150),
    batch_size = 20,
    class_mode = "binary"
)

model %>% compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_rmsprop(lr = 2e-5),
    metrics = c("accuracy")
)

# Highly CPU intensive - 15 mins per epoch
history <- model %>% fit_generator(
    train_generator,
    steps_per_epoch = 100,
    epochs = 2,
    validation_data = validation_generator,
    validation_steps = 50
)
