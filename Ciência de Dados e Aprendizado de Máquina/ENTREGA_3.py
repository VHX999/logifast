import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

data = pd.read_csv('./sample_data/database-logifast.csv')

df = pd.DataFrame(data)

bins = [0, 500, 1000, np.inf]  
labels = ["baixo", "médio", "alto"]  
df["CategoriaFrete"] = pd.cut(df["ValorFrete"], bins=bins, labels=labels)

X = df[["Km", "CombustIvel", "Manutencao", "CustosFixos"]]  
y = df["CategoriaFrete"]  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test)

print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("Matriz de Confusão")
plt.xlabel("Previsões")
plt.ylabel("Valores Reais")
plt.show()
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df["Km"], y=df["ValorFrete"], hue=df["CategoriaFrete"], palette="coolwarm")
plt.title("Classificação: Km vs ValorFrete (Categorias)")
plt.xlabel("Km")
plt.ylabel("ValorFrete")
plt.legend(title="Categoria de Frete")
plt.grid(True)
plt.show()
