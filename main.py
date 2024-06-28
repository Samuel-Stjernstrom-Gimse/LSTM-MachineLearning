
import json
import numpy as np
import pickle
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta

params = {
    'seq_length': 300,  # Sequence length for LSTM
    'lstm_units': 128,  # Number of LSTM units
    'epochs': 500,  # Number of epochs to train
    'batch_size': 64,  # Batch size for training
    'learning_rate': 0.001,  # Learning rate for the optimizer
    'patience': 25,  # Patience for early stopping
    'validation_split': 0.6,  # Fraction of data to use for validation
    'coin_id': 'bitcoin',
}

cryptocurrency_json_files = [
    "1inch.json",
    "aave.json",
    "aioz-network.json",
    "akash-network.json",
    "algorand.json",
    "aragon.json",
    "arweave.json",
    "avalanche.json",
    "axie-infinity.json",
    "binance-coin.json",
    "bitcoin.json",
    "bitcoin-cash.json",
    "bitcoin-sv.json",
    "cardano.json",
    "celo.json",
    "chainlink.json",
    "chiliz.json",
    "compound.json",
    "conflux-network.json",
    "cosmos.json",
    "crypto-com-coin.json",
    "curve-dao-token.json",
    "decentraland.json",
    "dexe.json",
    "dogecoin.json",
    "ecash.json",
    "elrond-egld.json",
    "eos.json",
    "ethereum.json",
    "ethereum-classic.json",
    "fantom.json",
    "fetch.json",
    "filecoin.json",
    "flow.json",
    "ftx-token.json",
    "gala.json",
    "gatetoken.json",
    "gnosis-gno.json",
    "golem-network-tokens.json",
    "hedera-hashgraph.json",
    "helium.json",
    "injective-protocol.json",
    "internet-computer.json",
    "iota.json",
    "iotex.json",
    "kava.json",
    "klaytn.json",
    "kucoin-token.json",
    "kusama.json",
    "lido-dao.json",
    "litecoin.json",
    "livepeer.json",
    "maker.json",
    "mantra-dao.json",
    "mina.json",
    "monero.json",
    "multi-collateral-dai.json",
    "near-protocol.json",
    "neo.json",
    "nervos-network.json",
    "nexo.json",
    "nxm.json",
    "oasis-network.json",
    "ocean-protocol.json",
    "okb.json",
    "pancakeswap.json",
    "pendle.json",
    "polkadot.json",
    "polygon.json",
    "quant.json",
    "raydium.json",
    "render-token.json",
    "shiba-inu.json",
    "singularitynet.json",
    "solana.json",
    "stacks.json",
    "stellar.json",
    "superfarm.json",
    "synthetix-network-token.json",
    "tether.json",
    "tezos.json",
    "the-graph.json",
    "the-sandbox.json",
    "theta.json",
    "theta-fuel.json",
    "thorchain.json",
    "tron.json",
    "trueusd.json",
    "trust-wallet-token.json",
    "uniswap.json",
    "unus-sed-leo.json",
    "usd-coin.json",
    "vechain.json",
    "wemix.json",
    "wootrade.json",
    "wrapped-bitcoin.json",
    "xinfin-network.json",
    "xrp.json",
    "zcash.json",
    "zilliqa.json"
]

print("Available cryptocurrency JSON files:")
for index, filename in enumerate(cryptocurrency_json_files, start=1):
    print(f"{index}. {filename[:-5]}")

# Get user input for file selection
while True:
    try:
        choice = int(input("Enter the number corresponding to the coin you want to choose: "))
        if 1 <= choice <= len(cryptocurrency_json_files):
            selected_file = cryptocurrency_json_files[choice - 1]
            break
        else:
            print("Invalid choice. Please enter a number within the range.")
    except ValueError:
        print("Invalid input. Please enter a number.")

# Extract the coin_id from the selected file
if selected_file.endswith('.json'):
    coin_id = selected_file[:-5]  # Remove '.json' to get the coin_id
else:
    coin_id = selected_file  # Fallback if the file name doesn't end with '.json'

# Set params['coin_id'] based on user's choice
params['coin_id'] = coin_id

# Output the selected coin_id
print(f"Selected coin_id: {params['coin_id']}")

# Function to load and preprocess data from multiple JSON files
def load_and_process_data(file_paths):
    all_data = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            data_dict = json.load(f)

        price_data = [(float(entry['priceUsd']), entry['time']) for entry in data_dict]
        all_data.extend(price_data)

    prices = [price for price, _ in all_data]

    # Normalize prices
    scaler = MinMaxScaler(feature_range=(0, 1))
    prices_normalized = scaler.fit_transform(np.array(prices).reshape(-1, 1))

    # Create sequences
    X, y = create_sequences(prices_normalized, params['seq_length'])
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=params['validation_split'], random_state=42)

    return X_train, X_val, y_train, y_val, scaler

# Function to create sequences
def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Define the LSTM model
def define_model(seq_length, lstm_units, learning_rate):
    model = Sequential()

    # LSTM layers
    model.add(LSTM(lstm_units, activation='relu', return_sequences=True, input_shape=(seq_length, 1)))
    model.add(Dropout(0.2))

    model.add(LSTM(lstm_units, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(lstm_units, activation='relu'))
    model.add(Dropout(0.2))

    # Dense layers
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))

    model.add(Dense(1))  # Output layer

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

    return model

# Function to save MinMaxScaler
def save_scaler(scaler, filename):
    with open(filename, 'wb') as f:
        pickle.dump(scaler, f)

# Function to load MinMaxScaler
def load_scaler(filename):
    with open(filename, 'rb') as f:
        scaler = pickle.load(f)
    return scaler

# List of file names (assuming these are cryptocurrency JSON files)
file_names = [f'cryptocurrency_data/{params["coin_id"]}.json']

# Training data from all files
X_train, X_val, y_train, y_val, scaler = load_and_process_data(file_names)

# Save the scaler for later use in prediction
save_scaler(scaler, 'scaler.pkl')

# Define the model with more layers
model = define_model(params['seq_length'], params['lstm_units'], params['learning_rate'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=params['patience'], restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=params['epochs'], batch_size=params['batch_size'],
                    validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=1)

# Save the trained model in native Keras format
save_model(model, f'saved_models/powerful_lstm_model_{params["coin_id"]}.keras')

print(f'Training completed. Model saved as "powerful_lstm_model_{params["coin_id"]}.keras" in native Keras format.')

# Example code for prediction using the trained model and saved scaler
def predict_with_model(model_path, scaler_path, historical_data_path, future_steps, seq_length):
    # Load the model
    model = load_model(model_path)

    # Extract the name from the historical_data_path
    name = historical_data_path.split('/')[-1].replace('.json', '')

    # Load the scaler
    scaler = load_scaler(scaler_path)

    # Load and preprocess historical data
    with open(historical_data_path, 'r') as f:
        historical_data = json.load(f)

    # Extract historical prices
    prices = [float(entry['priceUsd']) for entry in historical_data]

    # Normalize prices using the loaded scaler
    prices_normalized = scaler.transform(np.array(prices).reshape(-1, 1))

    # Create sequences for prediction
    X, _ = create_sequences(prices_normalized, seq_length)  # Unpack only X

    # Reshape X for model prediction
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Make predictions for the historical data
    predictions = model.predict(X)

    # Extend sequences for future predictions
    future_sequences = [X[-1]]  # Start with the last sequence from historical data

    for _ in range(future_steps):
        # Predict the next price
        next_price_normalized = model.predict(np.array([future_sequences[-1]]))
        future_sequences.append(np.append(future_sequences[-1][1:], next_price_normalized, axis=0))

    # Convert future sequences to numpy array
    future_sequences = np.array(future_sequences)

    # Make predictions using the loaded model for the future sequences
    future_predictions = model.predict(future_sequences)

    # Inverse transform the predictions to get actual prices
    future_predicted_prices = scaler.inverse_transform(future_predictions)

    # Print the predicted prices with the day in the future
    current_date = datetime.now()

    # Print the predicted prices with the day in the future
    print("Predicted Prices:")
    for day, price in enumerate(future_predicted_prices, start=1):
        future_date = current_date + timedelta(days=day)
        print(f"Price:{price[0]:.6f} Coin:{name}  Date:{future_date.strftime('%Y-%m-%d')}")

# Example usage for predicting future prices
historical_data_path = f'cryptocurrency_data/{params["coin_id"]}.json'
predict_with_model(f'saved_models/powerful_lstm_model_{params["coin_id"]}.keras', 'scaler.pkl', historical_data_path, future_steps=150, seq_length=params['seq_length'])

