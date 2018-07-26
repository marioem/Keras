library(keras)

model <- application_vgg16(weights = "imagenet")

img_path <- "~/Desktop/AG/sad-looking-yellow-lab-with-head-tilted-on-chair-back-in-the-pack-dog-portraits.jpg"
img <- image_load(img_path, target_size = c(224, 224)) %>%
image_to_array() %>%
array_reshape(dim = c(1, 224, 224, 3)) %>%
imagenet_preprocess_input()

preds <- model %>% predict(img)
imagenet_decode_predictions(preds, top = 3)[[1]]

max_act <- which.max(preds[1,])

picture_output <- model$output[, max_act]

# The following 3 assignments are kind of addressing the tensor of interest
# in order to later retrieve values of some of them: pooled_grads and last_conv_layer$output
#
last_conv_layer <- model %>% get_layer("block5_conv3") # Get the feature map of the last
                                                       # conv layer
grads <- k_gradients(picture_output, last_conv_layer$output)[[1]] # Gradient of the identified class wrt. to the last conv layer feature map
pooled_grads <- k_mean(grads, axis = c(1, 2, 3))

# Retrieve values of the two tesors in the second argument given the input in the first one
iterate <- k_function(list(model$input), list(pooled_grads, last_conv_layer$output[1,,,]))

c(pooled_grads_value, conv_layer_output_value) %<-% iterate(list(img))

for (i in 1:512) {
    conv_layer_output_value[,,i] <-    # Intensify feature map parts most important to the decided class
        conv_layer_output_value[,,i] * pooled_grads_value[[i]]
}

heatmap <- apply(conv_layer_output_value, c(1,2), mean) # Flatten (14,14,512) to (14,14) by averaging over all channels

heatmap <- pmax(heatmap, 0)
heatmap <- heatmap / max(heatmap)
write_heatmap <- function(heatmap, filename, width = 224, height = 224,
                          bg = "white", col = terrain.colors(12)) {
    png(filename, width = width, height = height, bg = bg)
    op = par(mar = c(0,0,0,0))
    on.exit({par(op); dev.off()}, add = TRUE)
    rotate <- function(x) t(apply(x, 2, rev))
    image(rotate(heatmap), axes = FALSE, asp = 1, col = col)
}
write_heatmap(heatmap, "elephant_heatmap.png")

library(magick)
library(viridis)
image <- image_read(img_path)
info <- image_info(image)
geometry <- sprintf("%dx%d!", info$width, info$height)
pal <- col2rgb(viridis(20), alpha = TRUE)
alpha <- floor(seq(0, 255, length = ncol(pal)))
pal_col <- rgb(t(pal), alpha = alpha, maxColorValue = 255)
write_heatmap(heatmap, "elephant_overlay.png",
              width = 14, height = 14, bg = NA, col = pal_col)
image_read("elephant_overlay.png") %>%
image_resize(geometry, filter = "quadratic") %>%
    image_composite(image, operator = "blend", compose_args = "20") %>%
    plot()
