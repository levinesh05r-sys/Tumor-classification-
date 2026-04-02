import numpy as np
# Monkeypatch for old tensorflowjs versions that use np.object
if not hasattr(np, 'object'):
    np.object = object
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float

import tensorflowjs as tfjs
import tensorflow as tf
import tf_keras as keras

model_path = r"d:\Tumor\keras_model.h5"
output_path = r"d:\Tumor\web_app\model"

print(f"Loading model from {model_path}...")
model = keras.models.load_model(model_path, compile=False)

print(f"Converting to TFJS Layers model at {output_path}...")
tfjs.converters.save_keras_model(model, output_path)
print("Conversion complete!")
