import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

# Cargar el archivo Excel
st.title("Análisis Interactivo de Estructuras Sedimentarias")

# Subir archivo Excel
uploaded_file = st.file_uploader("Cargar archivo Excel", type=["xlsx"])

if uploaded_file:
    # Leer el archivo Excel con Pandas
    df = pd.read_excel(uploaded_file)

    # Mostrar las primeras filas del DataFrame
    st.write("Vista previa de los datos:", df.head())

    # Mostrar los nombres de las columnas
    st.write("Columnas en el archivo:", df.columns)

    # 1. Análisis de frecuencia para columnas cualitativas (categorías)
    st.write("Distribución de 'Minerales principales'")
    mineral_counts = df['Minerales principales'].value_counts()
    st.write(mineral_counts)

    # Gráfico de barras para "Minerales principales"
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=mineral_counts.index, y=mineral_counts.values, ax=ax, palette='Set2')
    ax.set_title("Distribución de Minerales Principales")
    ax.set_xlabel("Mineral Principal")
    ax.set_ylabel("Frecuencia")
    st.pyplot(fig)  # Pasar el objeto fig aquí

    # 2. Análisis de frecuencia para 'Tipo de estructura'
    st.write("Distribución de 'Tipo de estructura'")
    structure_counts = df['Tipo de estructura'].value_counts()
    st.write(structure_counts)

    # Gráfico de barras para "Tipo de estructura"
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=structure_counts.index, y=structure_counts.values, ax=ax, palette='Set3')
    ax.set_title("Distribución de Tipo de Estructura")
    ax.set_xlabel("Tipo de Estructura")
    ax.set_ylabel("Frecuencia")
    st.pyplot(fig)  # Pasar el objeto fig aquí

    # 3. Codificación para regresión
    # Codificar "Grado de cementación" y "Tamaño del grano"
    cementacion_mapping = {"Mal": 1, "Regular": 2, "Bien": 3}
    df['Grado de cementación'] = df['Grado de cementación'].map(cementacion_mapping)

    grano_mapping = {"Arena": 1, "Grava": 2, "Limo": 3}
    df['Tamaño del grano'] = df['Tamaño del grano'].map(grano_mapping)

    # 4. Gráfico radar para comparar los parámetros seleccionados
    st.write("Comparación de múltiples parámetros entre las muestras seleccionadas")
    df_radar = df[['Porosidad (%)', 'Tamaño del grano', 'Edad geológica (Ma)', 'Grado de cementación']]

    # Eliminar filas con valores nulos (si hay)
    df_radar = df_radar.dropna()

    # Normalización
    df_radar_normalized = df_radar.apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    # Crear el gráfico Radar
    categories = df_radar_normalized.columns
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))  # Crear figura y ejes para el radar

    for index, row in df_radar_normalized.iterrows():
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, label=index)
        ax.fill(angles, values, alpha=0.25)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

    ax.set_title('Comparación de Parámetros entre Muestras')
    st.pyplot(fig)  # Pasar el objeto fig aquí
