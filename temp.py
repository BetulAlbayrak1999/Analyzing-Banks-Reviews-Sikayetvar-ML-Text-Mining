import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "data", "raw", "sikayetvar_isbank.csv")

df = pd.read_csv(file_path, encoding="utf-8")
print(df.head())
print(file_path)


