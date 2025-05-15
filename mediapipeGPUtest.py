import cv2
import numpy as np
import tensorflow as tf

# Load the GPU delegate
delegate = tf.lite.experimental.load_delegate(
    "/home/rasmus/tensorflow/bazel-bin/tensorflow/lite/delegates/gpu/libtensorflowlite_gpu_delegate.so"
)

# Load interpreter with GPU delegate
interpreter = tf.lite.Interpreter(model_path="your_model.tflite", experimental_delegates=[delegate])

try:
    interpreter.allocate_tensors()
except RuntimeError as e:
    print("Failed to delegate fully to GPU. Falling back to CPU. Error:", e)
    interpreter = tf.lite.Interpreter(model_path="your_model.tflite")  # reload without delegate
    interpreter.allocate_tensors()


input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()