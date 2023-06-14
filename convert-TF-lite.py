import tensorflow as tf
import os
#print(help(tf.lite.TFLiteConverter))

# Get the current working directory
current_directory = os.getcwd()

# Join the current directory with a subdirectory or filename
MODEL_DIR = os.path.join(current_directory, 'tf_model')

# Print the joined path
print(MODEL_DIR)

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR) # path to the SavedModel directory
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # Enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS,  # Enable TensorFlow ops.
]

converter.optimizations = [tf.lite.Optimize.DEFAULT]
#Quantizing weights

tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)