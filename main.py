


import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

params = {
    'seq_length': 100,         # Sequence length for LSTM
    'lstm_units': 128,         # Number of LSTM units
    'epochs': 1000,            # Number of epochs to train
    'batch_size': 32,          # Batch size for training
    'learning_rate': 0.001,    # Learning rate for the optimizer
    'patience': 15,            # Patience for early stopping
    'validation_split': 0.3    # Fraction of data to use for validation
}

# Function to load and preprocess data from multiple JSON files
def load_and_process_data(file_paths):
    all_data = []
    for file_path in file_paths:
        with open(f'cryptocurrency_data/{file_path}', 'r') as f:
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
    model.add(LSTM(lstm_units, activation='relu', input_shape=(seq_length, 1)))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model

# List of file names (assuming these are cryptocurrency IDs)
file_names = [
    '1inch.json', 'aave.json', 'aioz-network.json', 'akash-network.json', 'algorand.json',
    'aragon.json', 'arweave.json', 'avalanche.json', 'axie-infinity.json', 'binance-coin.json',
    'bitcoin.json', 'bitcoin-cash.json', 'bitcoin-sv.json', 'cardano.json', 'celo.json',
    'chainlink.json', 'chiliz.json', 'compound.json', 'conflux-network.json', 'cosmos.json',
    'crypto-com-coin.json', 'curve-dao-token.json', 'decentraland.json', 'dexe.json', 'dogecoin.json',
    'ecash.json', 'elrond-egld.json', 'eos.json', 'ethereum.json', 'ethereum-classic.json',
    'fantom.json', 'fetch.json', 'filecoin.json', 'flow.json', 'ftx-token.json', 'gala.json',
    'gatetoken.json', 'gnosis-gno.json', 'golem-network-tokens.json', 'hedera-hashgraph.json',
    'helium.json', 'injective-protocol.json', 'internet-computer.json', 'iota.json', 'iotex.json',
    'kava.json', 'klaytn.json', 'kucoin-token.json', 'kusama.json', 'lido-dao.json', 'litecoin.json',
    'livepeer.json', 'maker.json', 'mantra-dao.json', 'mina.json', 'monero.json', 'multi-collateral-dai.json',
    'near-protocol.json', 'neo.json', 'nervos-network.json', 'nexo.json', 'nxm.json', 'oasis-network.json',
    'ocean-protocol.json', 'okb.json', 'pancakeswap.json', 'pendle.json', 'polkadot.json', 'polygon.json',
    'quant.json', 'raydium.json', 'render-token.json', 'shiba-inu.json', 'singularitynet.json', 'solana.json',
    'stacks.json', 'stellar.json', 'superfarm.json', 'synthetix-network-token.json', 'tether.json', 'tezos.json',
    'the-graph.json', 'the-sandbox.json', 'theta.json', 'theta-fuel.json', 'thorchain.json', 'tron.json',
    'trueusd.json', 'trust-wallet-token.json', 'uniswap.json', 'unus-sed-leo.json', 'usd-coin.json', 'vechain.json',
    'wemix.json', 'wootrade.json', 'wrapped-bitcoin.json', 'xinfin-network.json', 'xrp.json', 'zcash.json', 'zilliqa.json'
]
# Training data from all files
X_train, X_val, y_train, y_val, scaler = load_and_process_data(file_names)

# Define the model
model = define_model(params['seq_length'], params['lstm_units'], params['learning_rate'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=params['patience'], restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=params['epochs'], batch_size=params['batch_size'],
                    validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=1)

# Save the trained model to HDF5 format
model.save('saved_models/powerful_lstm_model.h5')

print("Training completed. Model saved as 'powerful_lstm_model.h5'.")
