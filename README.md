import pandas as pd

# Cargar el dataset
df = pd.read_csv("heart_cleveland_upload.csv")

# Mostrar las primeras filas para revisar
df.head()

# Información general del dataset
df.info()

# Resumen estadístico
df.describe()

# Verificar si hay valores nulos
df.isnull().sum()

# Si hay valores nulos, eliminarlos
df.dropna(inplace=True)

# Si hay variables categóricas, convertirlas en numéricas
df = pd.get_dummies(df, drop_first=True)

# Definir X (todas las columnas menos la columna objetivo) e y (la variable objetivo)
X = df.drop("condition", axis=1)  # Cambia 'target' por el nombre correcto de la columna si es diferente
y = df["condition"]

from sklearn.model_selection import train_test_split

# Dividir los datos en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression

# Crear el modelo de Regresión Lineal
model = LinearRegression()

# Entrenar el modelo con los datos de entrenamiento
model.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error, r2_score

# Hacer predicciones con los datos de prueba
y_pred = model.predict(X_test)

# Evaluar el desempeño del modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

import matplotlib.pyplot as plt

# Graficar los valores predichos vs. los reales
plt.scatter(y_test, y_pred)
plt.xlabel("Valores Reales")
plt.ylabel("Valores Predichos")
plt.title("Regresión Lineal: Valores Reales vs. Predichos")
plt.show()
