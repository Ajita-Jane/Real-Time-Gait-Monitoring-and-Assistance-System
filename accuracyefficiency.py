import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import time

# Load the dataset
dataset_path = r"D:\Jane\1. College\8. Semester VIII\Auto Encoders\Normal1.xlsx" # Update the path if needed
df = pd.read_excel(dataset_path)

# Load the trained autoencoder model
model_path = r"D:\Jane\1. College\8. Semester VIII\Auto Encoders\autoencoder_model.h5" # Update the path if needed
autoencoder = tf.keras.models.load_model(model_path, custom_objects={"mse": tf.keras.losses.MeanSquaredError()})

# Load the scaler for preprocessing
scaler_path = r"D:\Jane\1. College\8. Semester VIII\Auto Encoders\scaler.pkl" # Update the path if needed
scaler = joblib.load(scaler_path)

# Preprocess the dataset
df_normalized = scaler.transform(df)

# Compute Reconstruction Error (MAE)
start_time = time.time()
reconstructed = autoencoder.predict(df_normalized)
end_time = time.time()

mae = np.mean(np.abs(df_normalized - reconstructed))

# Compute Inference Time
inference_time_per_sample = (end_time - start_time) / len(df_normalized) * 1000  # Convert to milliseconds

# Display results
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Average Inference Time per Sample: {inference_time_per_sample:.2f} ms")
