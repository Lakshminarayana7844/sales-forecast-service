import argparse
import json

from .model import train

p = argparse.ArgumentParser(prog="forecast")
p.add_argument("--data", default="data/weekly_sales.csv")
p.add_argument("--out", default="models")
a = p.parse_args()
print(json.dumps(train(a.data, a.out), indent=2))
