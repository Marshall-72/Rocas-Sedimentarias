import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

# Cargar el archivo Excel
st.title("Análisis Interactivo de Estructuras Sedimentarias")
# Mostrar los nombres de las columnas
st.write("Columnas en el archivo:", df.columns)
# Subir archivo Excel
uploaded_file = st.file_uploader("Cargar archivo Excel", type=["xlsx"])

if uploaded_file:
    # Leer el archivo Excel con Pandas
    df = pd.read_excel(uploaded_file)

    # Mostrar las primeras filas del DataFrame
    st.write("Vista previa de los datos:", df.head())

    # Estadísticas descriptivas
    st.write("Estadísticas descriptivas:", df.describe())

    # 1. Selección de muestras para comparación
    st.write("Selecciona las muestras que deseas comparar:")
    muestras_seleccionadas = st.multiselect('Muestras', df['Nombre Muestra'].unique())

    if muestras_seleccionadas:
        # Filtrar las muestras seleccionadas
        df_seleccionado = df[df['Nombre Muestra'].isin(muestras_seleccionadas)]

        # 2. Gráfico de barras para comparar el tamaño del grano de las muestras seleccionadas
        st.write("Comparación del Tamaño del Grano de las Muestras Seleccionadas")
        plt.figure(figsize=(8, 6))
        sns.barplot(x='Nombre Muestra', y='Tamaño del grano', data=df_seleccionado, palette='Set2')
        plt.title("Tamaño del Grano por Muestra")
        plt.xlabel("Muestra")
        plt.ylabel("Tamaño del Grano")
        st.pyplot()

        # 3. Gráfico de dispersión interactivo: Comparar Porosidad vs. Tamaño del grano
        st.write("Comparación entre Porosidad y Tamaño del Grano")
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='Porosidad (%)', y='Tamaño del grano', data=df_seleccionado, hue='Nombre Muestra', palette='Set2')
        plt.title("Porosidad vs Tamaño del Grano")
        plt.xlabel("Porosidad (%)")
        plt.ylabel("Tamaño del Grano")
        st.pyplot()

        # 4. Regresión lineal entre Porosidad y Grado de Cementación
        st.write("Relación entre Porosidad y Grado de Cementación")
        X = df_seleccionado[['Porosidad (%)']].values
        y = df_seleccionado['Grado de cementación'].astype('category').cat.codes.values  # Convertir a valores numéricos

        # Crear el modelo de regresión lineal
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        # Gráfico de dispersión con línea de regresión
        plt.figure(figsize=(8, 6))
        plt.scatter(X, y, color='blue')
        plt.plot(X, y_pred, color='red', linewidth=2)
        plt.title("Relación entre Porosidad y Grado de Cementación")
        plt.xlabel("Porosidad (%)")
        plt.ylabel("Grado de Cementación")
        st.pyplot()

        # 5. Gráfico Radar para comparar múltiples parámetros entre las muestras seleccionadas
        st.write("Comparación de múltiples parámetros entre las muestras seleccionadas")

        # Prepara los datos para el gráfico radar (porosidad, tamaño del grano, densidad, etc.)
        df_radar = df_seleccionado[['Nombre Muestra', 'Porosidad (%)', 'Tamaño del grano', 'Densidad aparente (g/cm³)', 'Edad geológica (Ma)']]
        df_radar = df_radar.set_index('Nombre Muestra')

        # Normalizamos los datos para el gráfico radar
        df_radar_normalized = df_radar.apply(lambda x: (x - x.min()) / (x.max() - x.min()))

        # Crear el gráfico Radar
        categories = df_radar_normalized.columns
        N = len(categories)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

        for index, row in df_radar_normalized.iterrows():
            values = row.tolist()
            values += values[:1]
            ax.plot(angles, values, label=index)
            ax.fill(angles, values, alpha=0.25)

        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

        plt.title('Comparación de Parámetros entre Muestras')
        st.pyplot(fig)


