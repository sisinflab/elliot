
import zipfile
import io
import requests
import os

from elliot.run import run_experiment

# url = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
# print(f"Getting Movielens 1Million from : {url} ..")
# response = requests.get(url)
#
# ml_1m_ratings = []
#
# print("Extracting ratings.dat ..")
# with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
#     for line in zip_ref.open("ml-1m/ratings.dat"):
#         ml_1m_ratings.append(str(line, "utf-8").replace("::", "\t"))
#
# print("Printing ratings.tsv to data/movielens_1m/ ..")
# os.makedirs("data/movielens_1m", exist_ok=True)
# with open("data/movielens_1m/dataset.tsv", "w") as f:
#     f.writelines(ml_1m_ratings)

# os.makedirs("data/movielens_1m", exist_ok=True)
# features = {}
# with open("data/movielens_1m/movies.tsv", "w") as fout:
#     with open("data/movielens_1m/movies.dat", "r", encoding='iso-8859-1') as fin:
#         for line in fin.readlines():
#             line_pattern = line.rstrip("\n").split("::")
#             item_id = line_pattern[0]
#             features_str = line_pattern[2].split("|")
#             features_int = []
#             for feature in features_str:
#                 features_int.append(str(features.setdefault(feature, len(features))))
#             feats_string = "\t".join(features_int)
#             fout.write(f"{item_id}\t{feats_string}\n")
#
# with open("data/movielens_1m/genres.tsv", "w") as f:
#     for genre, id in features.items():
#         f.write(f"{genre}\t{str(id)}\n")


print("Done! We are now starting the Elliot's experiment")
run_experiment("config_files/test_config_pe.yml")
