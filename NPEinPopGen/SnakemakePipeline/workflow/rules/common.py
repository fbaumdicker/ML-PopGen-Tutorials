# common.py

import pandas as pd

def count_scenario(p):
    df = pd.read_csv(p, index_col="scenario")
    return df.shape[0]
