import zipfile
import io
import requests
import os

from elliot.run import run_experiment

url = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
print(f"Getting Movielens 100K from : {url} ..")
response = requests.get(url)

ml_100k_ratings = []

print("Extracting ratings ..")
with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
    for line in zip_ref.open("ml-100k/u.data"):
        ml_100k_ratings.append(str(line, "utf-8"))

print("Printing ratings.tsv to data/movielens_100k/ ..")
os.makedirs("data/movielens_100k", exist_ok=True)
with open("data/movielens_100k/dataset.tsv", "w") as f:
    f.writelines(ml_100k_ratings)

print("Done! We are now starting the Elliot's experiment")
run_experiment("config_files/sample_gan.yml")