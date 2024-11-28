import pandas as pd

# Utilizar o pandas para ler a base.
data = pd.read_csv('./sample_data/database-logifast.csv')
df = pd.DataFrame(data)
# Ler a base
print(df)