import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def polynomial_regression(X, y, degree):
    """
    Realiza regresión de mínimos cuadrados sobre una base de polinomios en n variables.

    Parámetros:
    - X: ndarray de tamaño (m, n), donde m es el número de muestras y n el número de variables.
    - y: ndarray de tamaño (m,), valores objetivo.
    - degree: int, grado del polinomio a ajustar.

    Retorna:
    - model: modelo de regresión ajustado.
    - poly_features: transformador de características polinomiales.
    """
    # Expandimos X con términos polinomiales hasta el grado especificado
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly_features.fit_transform(X)

    # Ajustamos el modelo de regresión lineal sobre la base polinómica
    model = LinearRegression()
    model.fit(X_poly, y)

    return model, poly_features

# # --- Ejemplo de uso ---
# # Datos de entrada (2 variables)
# X = np.random.rand(100, 2) * 10  # 100 muestras, 2 variables
# y = 3 + 2 * X[:,0] + 3 * X[:,1]**2 + np.random.randn(100) * 2  # Función cuadrática con ruido

# # Ajustamos un polinomio de grado 2
# degree = 2
# model, poly_features = polynomial_regression(X, y, degree)

# # Predicción sobre nuevos datos
# X_new = np.array([[2, 3], [5, 1], [7, 8]])  # Nuevas muestras
# X_new_poly = poly_features.transform(X_new)
# y_pred = model.predict(X_new_poly)

# print("Predicciones:", y_pred)
