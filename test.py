import tensorflow as tf
import os

current_directory = os.getcwd()

# Join the current directory with a subdirectory or filename
MODEL_DIR = os.path.join(current_directory, 'tf_model')

# Load the model
model = tf.saved_model.load('tf_model')
print(model)

min_value = float('inf')
max_value = float('-inf')

for layer in model.layers:
    layer_weights = layer.get_weights()
    for weights in layer_weights:
        weights_min = tf.reduce_min(weights)
        weights_max = tf.reduce_max(weights)
        
        if weights_min < min_value:
            min_value = weights_min

        if weights_max > max_value:
            max_value = weights_max

print("Minimum weight value:", min_value.numpy())
print("Maximum weight value:", max_value.numpy())
