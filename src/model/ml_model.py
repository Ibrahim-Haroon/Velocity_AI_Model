import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load training dataset
train_df = pd.read_csv('training_dataset.csv')

# Extract features (X) and target variable (y)
X_train = train_df[['Min_Duration', 'Max_Duration']]
y_train = train_df['Task']

# Load validation dataset
val_df = pd.read_csv('validation_dataset.csv')

# Extract features (X) and target variable (y)
X_val = val_df[['Min_Duration', 'Max_Duration']]
y_val = val_df['Task']

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Build the neural network model
regression_model = Sequential()
regression_model.add(Dense(64, input_dim=2, activation='relu'))
regression_model.add(Dense(32, activation='relu'))
regression_model.add(Dense(1, activation='linear'))

# Compile model
regression_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

# Train model
regression_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model on the validation set
loss, mae = regression_model.evaluate(X_val, y_val)
print(f'Mean Absolute Error on Validation Set: {mae}')