# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1qOigN2szhPlwUOg9Mo-Ij2cCLUwUDUUk
"""

import pandas as pd

# Utilizar o pandas para ler a base.
data = pd.read_csv('./sample_data/LogisticData.csv')
# Organizar a Base
df = pd.DataFrame(data)
# Ler a base
print(df)