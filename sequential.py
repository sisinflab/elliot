from elliot.run import run_experiment
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('configs', type=str, nargs='+')

args = parser.parse_args()

for config in args.configs:
    if not os.path.exists(f"config_files/{config}.yml"):
        raise FileExistsError

for config in args.configs:
    run_experiment(f"config_files/{config}.yml")
