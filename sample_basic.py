
import zipfile
import io
import requests
import os

from elliot.run import run_experiment

url = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
print(f"Getting Movielens 1Million from : {url} ..")
response = requests.get(url)

ml_1m_ratings = []

print("Extracting ratings.dat ..")
with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
    for line in zip_ref.open("ml-1m/ratings.dat"):
        ml_1m_ratings.append(str(line, "utf-8").replace("::", "\t"))

print("Printing ratings.tsv to data/movielens_1m/ ..")

os.makedirs("data/movielens_1m", exist_ok=True)
with open("data/movielens_1m/dataset.tsv", "w") as f:
    f.writelines(ml_1m_ratings)

print("Done! We are now starting the Elliot's experiment")
run_experiment("config_files/basic_configuration_v030.yml")
