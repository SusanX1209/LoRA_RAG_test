

import pandas as pd
import json
# 256,916

data_path = './dataset/dialogues.parquet'
df = pd.read_parquet(data_path)
print(df.head())
