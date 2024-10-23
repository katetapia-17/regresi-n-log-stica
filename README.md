import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Cargar el dataset
df = pd.read_csv("heart_cleveland_upload.csv")

# Definir las variables predictoras (X) y la variable objetivo (y)
X = df.drop("condition", axis=1)
y = df["condition"]

# Dividir el dataset en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Crear el modelo de Regresión Logística
model = LogisticRegression(max_iter=200)  # Aumentar el número de iteraciones
model.fit(X_train_scaled, y_train)

# Realizar predicciones
y_pred = model.predict(X_test_scaled)

# Evaluar el modelo
print(f"Precisión: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Obtener las probabilidades predichas para la clase 1
y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]

# Calcular la curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Calcular el AUC (Área bajo la curva)
auc_score = roc_auc_score(y_test, y_pred_prob)

# Graficar la curva ROC
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC para el Modelo de Regresión Logística')
plt.legend(loc='best')
plt.grid(True)
plt.show()
