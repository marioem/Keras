library(keras)

model <- application_vgg16(
    weights = "imagenet",
    include_top = FALSE
    )
layer_name <- "block3_conv1"
filter_index <- 1
layer_output <- get_layer(model, layer_name)$output
loss <- k_mean(layer_output[,,,filter_index])

grads <- k_gradients(loss, model$input)[[1]]
#gradient normalization trick
grads <- grads / (k_sqrt(k_mean(k_square(grads))) + 1e-5)

# Get the computed loss and gradient tensors given the input image
iterate <- k_function(list(model$input), list(loss, grads))
c(loss_value, grads_value) %<-%
    iterate(list(array(0, dim = c(1, 150, 150, 3))))

input_img_data <- array(runif(150 * 150 * 3), dim = c(1, 150, 150, 3)) * 20 + 128
step <- 1
for (i in 1:40) {
    c(loss_value, grads_value) %<-% iterate(list(input_img_data))
    input_img_data <- input_img_data + (grads_value * step)
}

deprocess_image <- function(x) {
    dms <- dim(x)
    x <- x - mean(x)            # Tensor normalization to mean = 0
    x <- x / (sd(x) + 1e-5)     # and std = .1
    x <- x * 0.1                #
    x <- x + 0.5
    x <- pmax(0, pmin(x, 1))    # clipping to [0,1]
    array(x, dim = dms)
}


# Putting all the above together
generate_pattern <- function(layer_name, filter_index, size = 150) {
    layer_output <- model$get_layer(layer_name)$output          # Build the loss function...
    loss <- k_mean(layer_output[,,,filter_index])               # that maximizes activation of the filter under consideration
    grads <- k_gradients(loss, model$input)[[1]]                # Computes the gradient of the input picture wrt to that loss
    grads <- grads / (k_sqrt(k_mean(k_square(grads))) + 1e-5)   # Normalize the gradient
    iterate <- k_function(list(model$input), list(loss, grads)) # Return the loss and gradient given the picture
    input_img_data <-                                           # Prepare initial gray sligtly noisy image
    array(runif(size * size * 3), dim = c(1, size, size, 3)) * 20 + 128
    step <- 1
    for (i in 1:40) {                                           # Run gradient ascent for 40 steps
        c(loss_value, grads_value) %<-% iterate(list(input_img_data))
        input_img_data <- input_img_data + (grads_value * step)
    }
    img <- input_img_data[1,,,]
    deprocess_image(img)
}

library(grid)
grid.raster(generate_pattern("block3_conv1", 1))

library(gridExtra)
dir.create("vgg_filters")
for (layer_name in c("block1_conv1", "block2_conv1",
                     "block3_conv1", "block4_conv1")) {
    size <- 140
    png(paste0("vgg_filters/", layer_name, ".png"),
         width = 8 * size, height = 8 * size)
    
    grobs <- list()
    for (i in 0:7) {
        for (j in 0:7) {
            pattern <- generate_pattern(layer_name, i + (j*8) + 1, size = size)
            grob <- rasterGrob(pattern,
                               width = unit(0.9, "npc"),
                               height = unit(0.9, "npc"))
            grobs[[length(grobs)+1]] <- grob
        }
    }
    
    grid.arrange(grobs = grobs, ncol = 8)
    dev.off()
}

