original_dataset_dir <- "~/Documents/GitHub/Keras/Cats and dogs/train/"
base_dir <- "~/Documents/GitHub/Keras/Cats and dogs small/"
# dir.create(base_dir)
train_dir <- file.path(base_dir, "train")
dir.create(train_dir)
validation_dir <- file.path(base_dir, "validation")
dir.create(validation_dir)
test_dir <- file.path(base_dir, "test")
dir.create(test_dir)
train_cats_dir <- file.path(train_dir, "cats")
dir.create(train_cats_dir)
train_dogs_dir <- file.path(train_dir, "dogs")
dir.create(train_dogs_dir)
validation_cats_dir <- file.path(validation_dir, "cats")
dir.create(validation_cats_dir)
validation_dogs_dir <- file.path(validation_dir, "dogs")
dir.create(validation_dogs_dir)
test_cats_dir <- file.path(test_dir, "cats")
dir.create(test_cats_dir)
test_dogs_dir <- file.path(test_dir, "dogs")
dir.create(test_dogs_dir)
fnames <- paste0("cat.", 1:1000, ".jpg")
file.copy(file.path(original_dataset_dir, fnames),
          file.path(train_cats_dir))
fnames <- paste0("cat.", 1001:1500, ".jpg")
file.copy(file.path(original_dataset_dir, fnames),
          file.path(validation_cats_dir))
fnames <- paste0("cat.", 1501:2000, ".jpg")
file.copy(file.path(original_dataset_dir, fnames),
          file.path(test_cats_dir))
fnames <- paste0("dog.", 1:1000, ".jpg")
file.copy(file.path(original_dataset_dir, fnames),
          file.path(train_dogs_dir))
fnames <- paste0("dog.", 1001:1500, ".jpg")
file.copy(file.path(original_dataset_dir, fnames),
          file.path(validation_dogs_dir))
fnames <- paste0("dog.", 1501:2000, ".jpg")
file.copy(file.path(original_dataset_dir, fnames),
          file.path(test_dogs_dir))

# cat("total training cat images:", length(list.files(train_cats_dir)), "\n")
# cat("total training dog images:", length(list.files(train_dogs_dir)), "\n")
# cat("total validation cat images:",
#       length(list.files(validation_cats_dir)), "\n")
# cat("total validation dog images:",
#       length(list.files(validation_dogs_dir)), "\n")
# cat("total test cat images:", length(list.files(test_cats_dir)), "\n")
# cat("total test dog images:", length(list.files(test_dogs_dir)), "\n")

library(keras)

train_datagen <- image_data_generator(rescale = 1/255)
validation_datagen <- image_data_generator(rescale = 1/255)

train_generator <- flow_images_from_directory(
    train_dir,
    train_datagen,
    target_size = c(150, 150),
    batch_size = 20,
    class_mode = "binary"
)

validation_generator <- flow_images_from_directory(
    validation_dir,
    validation_datagen,
    target_size = c(150, 150),
    batch_size = 20,
    class_mode = "binary"
)

batch <- generator_next(train_generator)
str(batch)


model <- keras_model_sequential() %>%
    layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                  input_shape = c(150, 150, 3)) %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_flatten() %>%
    layer_dense(units = 512, activation = "relu") %>%
    layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_rmsprop(lr = 1e-4),
    metrics = c("acc")
)

history <- model %>% fit_generator(
    train_generator,
    steps_per_epoch = 100,         # of batches of 20 from 2000 images
    epochs = 30,
    validation_data = validation_generator,
    validation_steps = 50          # of batches of 20 from 1000 images
)

model %>% save_model_hdf5("cats_and_dogs_small_1.h5")

################################################################################
#
# Data augmentation
#
datagen <- image_data_generator(
    rescale = 1/255,
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = TRUE,
    fill_mode = "nearest"
)

fnames <- list.files(train_cats_dir, full.names = TRUE)
img_path <- fnames[[7]]

img <- image_load(img_path, target_size = c(150, 150))
img_array <- image_to_array(img)
img_array <- array_reshape(img_array, c(1, 150, 150, 3))

# Purely for visualization of some example pictures

augmentation_generator <- flow_images_from_data(
                                                img_array,
                                                generator = datagen,
                                                batch_size = 1
)

op <- par(mfrow = c(2, 2), pty = "s", mar = c(1, 0, 1, 0))
for (i in 1:4) {
    batch <- generator_next(augmentation_generator)
    plot(as.raster(batch[1,,,]))
}
par(op)

# Generation for training a new model with data augmentation and dropouts

test_datagen <- image_data_generator(rescale = 1/255)
train_generator <- flow_images_from_directory(
    train_dir,
    datagen,
    target_size = c(150, 150),
    batch_size = 32,
    class_mode = "binary"
)

validation_generator <- flow_images_from_directory(
    validation_dir,
    test_datagen,
    target_size = c(150, 150),
    batch_size = 32,
    class_mode = "binary"
)

model2 <- keras_model_sequential() %>%
    layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                  input_shape = c(150, 150, 3)) %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%

    layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%

    layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    
    layer_flatten() %>%
    layer_dropout(rate = 0.5) %>%
    layer_dense(units = 512, activation = "relu") %>%
    layer_dense(units = 1, activation = "sigmoid")

model2 %>% compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_rmsprop(lr = 1e-4),
    metrics = c("acc")
)

history <- model2 %>% fit_generator(
    train_generator,
    steps_per_epoch = 100,
    epochs = 100,
    validation_data = validation_generator,
    validation_steps = 50       # of batches of 32 from 1000 images
)

model2 %>% save_model_hdf5("cats_and_dogs_small_2.h5")

