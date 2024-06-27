import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Adjustable parameters
params = {
    'seq_length': 10,         # Sequence length for LSTM
    'lstm_units': 100,        # Number of LSTM units
    'epochs': 200,           # Number of epochs to train
    'batch_size': 32,         # Batch size for training
    'learning_rate': 0.001,   # Learning rate for the optimizer
    'patience': 20,           # Patience for early stopping
    'validation_split': 0.2   # Fraction of data to use for validation
}

# Load JSON data from file
with open('price_data.json', 'r') as f:
    data_dict = json.load(f)

# Extract priceUsd and time from data
price_data = [(float(entry['priceUsd']), entry['time']) for entry in data_dict['data']]

# Extract prices and times separately
prices = [price for price, _ in price_data]

# Normalize the prices
scaler = MinMaxScaler(feature_range=(0, 1))
prices_normalized = scaler.fit_transform(np.array(prices).reshape(-1, 1))

# Function to create sequences for LSTM
def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Create sequences
X, y = create_sequences(prices_normalized, params['seq_length'])

# Reshape for LSTM input (samples, time steps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=params['validation_split'], random_state=42)

# Build the LSTM model
model = Sequential()
model.add(LSTM(params['lstm_units'], activation='relu', input_shape=(params['seq_length'], 1)))
model.add(Dense(1))
model.compile(optimizer=Adam(learning_rate=params['learning_rate']), loss='mse')

# Implement early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=params['patience'], restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=params['epochs'], batch_size=params['batch_size'], validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=1)

# Make predictions for the next 100 days
predicted_prices = []
current_sequence = X[-1]  # Start with the last sequence from the data

for _ in range(100):
    # Reshape current_sequence for prediction
    current_sequence = current_sequence.reshape((1, params['seq_length'], 1))

    # Predict the next price
    predicted_price_normalized = model.predict(current_sequence)[0][0]

    # Store the predicted price
    predicted_prices.append(predicted_price_normalized)

    # Reshape predicted_price_normalized to (1, 1, 1) to match current_sequence shape
    predicted_price_normalized = np.array([[[predicted_price_normalized]]])

    # Update current_sequence to include the predicted price and drop the first element
    current_sequence = np.append(current_sequence[:, 1:, :], predicted_price_normalized, axis=1)

# Inverse transform to get the predicted prices in original scale
predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))

# Display predicted prices for the next 100 days
for i, price in enumerate(predicted_prices, start=1):
    print(f"Day {i}: Predicted price = {price[0]}")
