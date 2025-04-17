import os, json
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi


# kaggle_dir = os.path.expanduser('~/.kaggle')
# if not os.path.exists(kaggle_dir):
#     os.makedirs(kaggle_dir)

# # Complete the JSON with both username and key
# with open(f"{kaggle_dir}/kaggle.json", "w") as f:
#     json.dump({
#         "username": "boluolumodeji",
#         "key": "e2a2f15dcf33ad52d6b0bd6ed4a662bc"  # Replace with your actual API key
#     }, f)

# # os.chmod(f"{kaggle_dir}/kaggle.json", 0o600)
api = KaggleApi()
api.authenticate()
dataset_owner = "asaniczka"
dataset_name = "tmdb-movies-dataset-2023-930k-movies"
download_path = "Users/Bolu/Movie_recommender"

if not os.path.exists(download_path):
    os.makedirs(download_path)
    
print(f"Downloading dataset {dataset_owner}/{dataset_name}")
api.dataset_download_files(
    dataset= f"{dataset_owner}/{dataset_name}",
    path= download_path,
    unzip=True
)