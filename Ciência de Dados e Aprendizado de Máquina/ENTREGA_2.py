import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd

# Importar os dados
data = pd.read_csv('./sample_data/database-logifast.csv')
df = pd.DataFrame(data)

correlation = df["Km"].corr(df["ValorFrete"])
X = df[["Km"]].values
y = df["ValorFrete"].values

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

print(f'Coeficiente angular (m): {model.coef_[0]:.4f}')
print(f'Intercepto (b): {model.intercept_:.4f}')

plt.figure(figsize=(10, 6))
plt.scatter(df["Km"], df["ValorFrete"], color='blue', label='Dados reais')  # Pontos reais
plt.plot(df["Km"], y_pred, color='red', label='Regressão Linear')  # Linha de regressão
plt.title('Regressão Linear: Km vs ValorFrete')
plt.xlabel('Km')
plt.ylabel('ValorFrete')
plt.legend()
plt.grid(True)
plt.show()