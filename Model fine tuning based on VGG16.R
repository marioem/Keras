################################################################################
# 
# Model fine-tuning:
# Similar to feature extraction Approach 2. but some of the top layers of the convolutional base get unfrozen and traines together with the classifier layres. Steps in the process:
#    a. Add your custom network on top of an already-trained base network.
#    b. Freeze the base network.
#    c. Train the part you added.
#    d. Unfreeze some layers in the base network.
#    e. Jointly train both these layers and the part you added.


conv_base <- application_vgg16(
    weights = "imagenet",
    include_top = FALSE,
    input_shape = c(150, 150, 3)
)

# a

model <- keras_model_sequential() %>%
    conv_base %>%
    layer_flatten() %>%
    layer_dense(units = 256, activation = "relu") %>%
    layer_dense(units = 1, activation = "sigmoid")

# Before you compile and train the model, it’s very important to freeze 
# the convolutional base.

cat("This is the number of trainable weights before freezing",
    "the conv base:", length(model$trainable_weights), "\n")

# b

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

# c
# Highly CPU intensive - 15 mins per epoch
history <- model %>% fit_generator(
    train_generator,
    steps_per_epoch = 100,
    epochs = 2,
    validation_data = validation_generator,
    validation_steps = 50
)

# d

unfreeze_weights(conv_base, from = "block3_conv1")

# e

model %>% compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_rmsprop(lr = 1e-5),
    metrics = c("accuracy")
)

history <- model %>% fit_generator(
    train_generator,
    steps_per_epoch = 100,
    epochs = 100,
    validation_data = validation_generator,
    validation_steps = 50
)

# Model evaluation

test_generator <- flow_images_from_directory(
    test_dir,
    test_datagen,
    target_size = c(150, 150),
    batch_size = 20,
    class_mode = "binary"
)

model %>% evaluate_generator(test_generator, steps = 50)

# Wrapping up
# - Convnets are the best type of machine-learning models for computer-vision 
# tasks. It’s possible to train one from scratch even on a very small dataset, 
# with decent results.
# - On a small dataset, overfitting will be the main issue. Data augmentation is 
# a powerful way to fight overfitting when you’re working with image data.
# - It’s easy to reuse an existing convnet on a new dataset via feature extraction.
# This is a valuable technique for working with small image datasets.
# - As a complement to feature extraction, you can use fine-tuning, which adapts 
# to a new problem some of the representations previously learned by an existing 
# model. This pushes performance a bit further.
