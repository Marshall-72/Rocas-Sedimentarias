import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Cargar los datos desde el archivo Excel
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

    # Gráfico simple: Espesor vs Nivel de Bioturbación
    if 'Espesor (cm)' in df.columns and 'Nivel de bioturbación (0–3)' in df.columns:
        st.write("Gráfico: Espesor vs Nivel de Bioturbación")
        plt.figure(figsize=(8, 6))
        plt.scatter(df['Espesor (cm)'], df['Nivel de bioturbación (0–3)'], color='blue')
        plt.title('Relación entre Espesor y Bioturbación')
        plt.xlabel('Espesor (cm)')
        plt.ylabel('Nivel de Bioturbación (0-3)')
        st.pyplot()
