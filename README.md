

```markdown
# Currency Price Prediction using LSTM

This repository contains a Long Short-Term Memory (LSTM) neural network model for predicting currency prices. The model is built using TensorFlow and Keras, and is trained on historical currency price data to make future predictions.

## Project Structure

- `price_data.json`: JSON file containing the historical currency price data.
- `currency_price_prediction.py`: Python script implementing the LSTM model and the prediction process.
- `README.md`: This file, providing an overview and instructions for the project.

## Requirements

- Python 3.x
- TensorFlow
- Keras
- scikit-learn
- numpy
- json

You can install the required libraries using the following command:

```sh
pip install tensorflow keras scikit-learn numpy
```

## Model Parameters

The model uses the following parameters:

- `seq_length`: 100 (Sequence length for LSTM)
- `lstm_units`: 128 (Number of LSTM units)
- `epochs`: 1000 (Number of epochs to train)
- `batch_size`: 32 (Batch size for training)
- `learning_rate`: 0.001 (Learning rate for the optimizer)
- `patience`: 15 (Patience for early stopping)
- `validation_split`: 0.3 (Fraction of data to use for validation)

## Data Preparation

The historical currency price data is stored in `price_data.json`. Each entry in the JSON file contains the price and the timestamp.

```json
{
    "data": [
        {"priceUsd": "0.1", "time": "2023-01-01T00:00:00Z"},
        {"priceUsd": "0.2", "time": "2023-01-02T00:00:00Z"},
        ...
    ]
}
```

## Running the Model

1. Load the data from `price_data.json`:

```python
with open('price_data.json', 'r') as f:
    data_dict = json.load(f)
```

2. Extract and normalize the prices:

```python
price_data = [(float(entry['priceUsd']), entry['time']) for entry in data_dict['data']]
prices = [price for price, _ in price_data]
scaler = MinMaxScaler(feature_range=(0, 1))
prices_normalized = scaler.fit_transform(np.array(prices).reshape(-1, 1))
```

3. Create sequences for the LSTM model:

```python
def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

X, y = create_sequences(prices_normalized, params['seq_length'])
X = X.reshape((X.shape[0], X.shape[1], 1))
```

4. Split the data into training and validation sets:

```python
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=params['validation_split'], random_state=42)
```

5. Build and train the LSTM model:

```python
model = Sequential()
model.add(LSTM(params['lstm_units'], activation='relu', input_shape=(params['seq_length'], 1)))
model.add(Dense(1))
model.compile(optimizer=Adam(learning_rate=params['learning_rate']), loss='mse')

early_stopping = EarlyStopping(monitor='val_loss', patience=params['patience'], restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=params['epochs'], batch_size=params['batch_size'], validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=1)
```

6. Make future predictions:

```python
predicted_prices = []
current_sequence = X[-1]

for _ in range(500):
    current_sequence = current_sequence.reshape((1, params['seq_length'], 1))
    predicted_price_normalized = model.predict(current_sequence)[0][0]
    predicted_prices.append(predicted_price_normalized)
    predicted_price_normalized = np.array([[[predicted_price_normalized]]])
    current_sequence = np.append(current_sequence[:, 1:, :], predicted_price_normalized, axis=1)

predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))

for i, price in enumerate(predicted_prices, start=1):
    print(f"Day {i}: Predicted price = {price[0]}")
```

## Results

The model's predictions for the next 500 days are printed to the console. The predicted prices are scaled back to their original values using the inverse transform of the scaler.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

This project uses the TensorFlow and Keras libraries for building and training the LSTM model, and scikit-learn for data preprocessing.
```


