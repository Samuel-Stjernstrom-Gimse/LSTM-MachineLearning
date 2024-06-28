import os
import requests
import json

# Function to fetch cryptocurrency IDs from API
def fetch_crypto_ids():
    url = 'https://api.coincap.io/v2/assets'
    response = requests.get(url)
    data = response.json()
    ids = [entry['id'] for entry in data['data']]
    return ids

# Function to fetch historical data for a given cryptocurrency ID
def fetch_historical_data(crypto_id):
    url = f'https://api.coincap.io/v2/assets/{crypto_id}/history?interval=d1'
    response = requests.get(url)
    data = response.json()
    return data['data']

# Function to save data to a JSON file
def save_to_file(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

# Main function to orchestrate fetching and saving
def main():
    # Create a folder to store data files if it doesn't exist
    folder_name = 'cryptocurrency_data'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Fetch cryptocurrency IDs
    crypto_ids = fetch_crypto_ids()

    # Fetch and save historical data for each cryptocurrency
    for crypto_id in crypto_ids:
        print(f"Fetching data for {crypto_id}...")
        historical_data = fetch_historical_data(crypto_id)
        file_path = os.path.join(folder_name, f"{crypto_id}.json")
        save_to_file(historical_data, file_path)
        print(f"Saved data for {crypto_id} to {file_path}")

if __name__ == "__main__":
    main()