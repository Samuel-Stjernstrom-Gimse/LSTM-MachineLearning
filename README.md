This repository contains code for training LSTM models to predict future prices of various cryptocurrencies based on historical data. The project utilizes TensorFlow and Python for model training and prediction.

Project Structure
main.py: Main script to fetch historical data, train LSTM models, and predict future prices.
cryptocurrency_data/: Directory to store JSON files containing historical price data for each cryptocurrency.
saved_models/: Directory to store trained LSTM models.
scaler.pkl: Pickle file containing the scaler used to normalize data during training.
Requirements
Ensure you have Python 3.x installed along with the following libraries:

numpy
tensorflow
scikit-learn
requests
Install dependencies using:

bash
Kopier kode
pip install -r requirements.txt
Setup
Clone the repository:

bash
Kopier kode
git clone https://github.com/your_username/cryptocurrency-lstm-prediction.git
cd cryptocurrency-lstm-prediction
Fetch cryptocurrency data:

Run the main script to fetch historical data for various cryptocurrencies:

bash
Kopier kode
python main.py
This will populate the cryptocurrency_data/ directory with JSON files for each cryptocurrency.

Train LSTM models:

Modify parameters in main.py such as seq_length, lstm_units, and epochs as needed. Then, run:

bash
Kopier kode
python main.py
LSTM models will be trained for the selected cryptocurrency and saved in saved_models/.

Predict future prices:

Use the trained models to predict future prices. Example usage:

bash
Kopier kode
python main.py
Adjust future_steps parameter to specify the number of days to predict into the future.

Usage
Select a cryptocurrency:

When prompted, choose a cryptocurrency from the list provided.

Prediction:

After training, predictions for future prices will be displayed based on the selected cryptocurrency and model.

Contributing
Feel free to contribute to this project by forking and submitting a pull request. Ensure code changes align with the repository's objectives.

Credits
Author: Your Name
Email: your.email@example.com
License
This project is licensed under the MIT License - see the LICENSE file for details.
