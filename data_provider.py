import pandas as pd
import numpy as np

df = pd.read_csv('metadata.csv', header=None)
print(df.values[0])
print(df.values[505])
print(df.describe())


