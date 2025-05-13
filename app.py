import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

# Cargar el archivo Excel
st.title("Análisis de Estructuras Sedimentarias")

# Subir archivo Excel
uploaded_file = st.file_uploader("Cargar archivo Excel", type=["xlsx"])

if uploaded_file:
    # Leer el archivo Excel con Pandas
    df = pd.read_excel(uploaded_file)

    # Mostrar las primeras filas del DataFrame
    st.write("Vista previa de los datos:", df.head())

    # Estadísticas descriptivas
    st.write("Estadísticas descriptivas:", df.describe())

    # 1. Gráfico de distribución del espesor
    st.write("Distribución del espesor de las muestras")
    plt.figure(figsize=(8, 6))
    sns.histplot(df['Espesor (cm)'], kde=True, bins=10, color='skyblue')
    plt.title("Distribución del espesor")
    plt.xlabel("Espesor (cm)")
    plt.ylabel("Frecuencia")
    st.pyplot()

    # 2. Gráfico de distribución de la porosidad
    st.write("Distribución de la porosidad de las muestras")
    plt.figure(figsize=(8, 6))
    sns.histplot(df['Porosidad (%)'], kde=True, bins=10, color='lightgreen')
    plt.title("Distribución de la Porosidad")
    plt.xlabel("Porosidad (%)")
    plt.ylabel("Frecuencia")
    st.pyplot()

    # 3. Gráfico de dispersión (espesor vs porosidad) con regresión lineal
    st.write("Relación entre espesor y porosidad")
    plt.figure(figsize=(8, 6))
    X = df[['Espesor (cm)']].values
    y = df['Porosidad (%)'].values

    # Crear el modelo de regresión lineal
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # Gráfico de dispersión con línea de regresión
    plt.scatter(X, y, color='blue')
    plt.plot(X, y_pred, color='red', linewidth=2)
    plt.title("Relación entre Espesor y Porosidad")
    plt.xlabel("Espesor (cm)")
    plt.ylabel("Porosidad (%)")
    st.pyplot()

    # 4. Boxplot para espesor y porosidad
    st.write("Boxplot de espesor y porosidad")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Boxplot de espesor
    sns.boxplot(x=df['Espesor (cm)'], ax=axes[0], color='lightgreen')
    axes[0].set_title("Boxplot de Espesor (cm)")

    # Boxplot de porosidad
    sns.boxplot(x=df['Porosidad (%)'], ax=axes[1], color='lightcoral')
    axes[1].set_title("Boxplot de Porosidad")

    st.pyplot(fig)
