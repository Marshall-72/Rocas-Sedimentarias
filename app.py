import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Visualización y filtro de estructuras sedimentarias")

# Paso 1: subir archivo Excel
uploaded_file = st.file_uploader("Sube tu archivo Excel corregido", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("Datos cargados:")
    st.dataframe(df)

    # Paso 2: filtros para cada columna
    filtros = {}
    for col in df.columns:
        if df[col].dtype == object:
            opciones = df[col].dropna().unique()
            seleccion = st.multiselect(f"Filtrar por {col}", opciones, default=opciones)
            filtros[col] = seleccion
        else:
            minimo = float(df[col].min())
            maximo = float(df[col].max())
            rango = st.slider(f"Rango para {col}", minimo, maximo, (minimo, maximo))
            filtros[col] = rango

    # Aplicar filtros
    df_filtrado = df.copy()
    for col, val in filtros.items():
        if isinstance(val, list):
            df_filtrado = df_filtrado[df_filtrado[col].isin(val)]
        else:
            df_filtrado = df_filtrado[(df_filtrado[col] >= val[0]) & (df_filtrado[col] <= val[1])]

    st.write("Datos filtrados:")
    st.dataframe(df_filtrado)

    # Sección de gráficos
    st.header("Gráficos")

    columnas = list(df_filtrado.columns)

    # Selección tipo gráfico
    tipo_grafico = st.selectbox("Selecciona tipo de gráfico", ["Barra", "Pastel", "Heatmap (2 columnas)"])

    if tipo_grafico in ["Barra", "Pastel"]:
        columna = st.selectbox("Selecciona la columna para graficar", columnas)
        data_graf = df_filtrado[columna].value_counts()

        fig, ax = plt.subplots()
        if tipo_grafico == "Barra":
            data_graf.plot(kind="bar", ax=ax)
            ax.set_ylabel("Frecuencia")
            ax.set_xlabel(columna)
            ax.set_title(f"Gráfico de barras - {columna}")
        else:  # Pastel
            data_graf.plot(kind="pie", autopct="%1.1f%%", ax=ax)
            ax.set_ylabel("")
            ax.set_title(f"Gráfico de pastel - {columna}")

        st.pyplot(fig)

    else:  # Heatmap
        col1 = st.selectbox("Selecciona columna 1", columnas, key="col1")
        col2 = st.selectbox("Selecciona columna 2", columnas, key="col2")

        # Crear tabla cruzada
        tabla_cruzada = pd.crosstab(df_filtrado[col1], df_filtrado[col2])

        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(tabla_cruzada, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
        ax.set_title(f"Heatmap de frecuencias entre {col1} y {col2}")

        st.pyplot(fig)

