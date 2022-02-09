from elliot.run import run_experiment
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('configs', type=str, nargs='+')

args = parser.parse_args()

for config in args.configs:
    run_experiment(f"config_files/{config}.yml")
