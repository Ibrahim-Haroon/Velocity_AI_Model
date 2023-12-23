import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load the CSV files into pandas dataframes
training_data = pd.read_csv('training_data.csv')
validation_data = pd.read_csv('validation_data.csv')

# Convert the dataframes into numpy arrays
training_data = training_data.to_numpy()
validation_data = validation_data.to_numpy()

# Split the data into features and labels
training_features = training_data[:, 1:]
training_labels = training_data[:, 0]
validation_features = validation_data[:, 1:]
validation_labels = validation_data[:, 0]

# Normalize the data
mean = training_features.mean(axis=0)
std = training_features.std(axis=0)
training_features = (training_features - mean) / std
validation_features = (validation_features - mean) / std

# Define the model architecture
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[training_features.shape[1]]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

# Compile the model
model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae', 'mse']
)

# Train the model
history = model.fit(
    training_features, training_labels,
    validation_data=(validation_features, validation_labels),
    batch_size=32,
    epochs=100
)
