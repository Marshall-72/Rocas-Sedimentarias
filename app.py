import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Visualización y filtro de estructuras sedimentarias")

# Paso 1: subir archivo Excel
uploaded_file = st.file_uploader("Sube tu archivo Excel", type=["xlsx"])
if uploaded_file:
    # Cargar datos
    df = pd.read_excel(uploaded_file)

    st.write("Datos cargados:")
    st.dataframe(df)

    # Paso 2: filtros para cada columna
    filtros = {}
    for col in df.columns:
        # Si la columna es categórica o texto, usar multiselección
        if df[col].dtype == object:
            opciones = df[col].dropna().unique()
            seleccion = st.multiselect(f"Filtrar por {col}", opciones, default=opciones)
            filtros[col] = seleccion
        else:
            # Si es numérica, usar slider para rango
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

    # Paso 3: generar gráfica
    st.subheader("Gráfica de tamaño de grano (ejemplo)")

    if "Tamaño de grano" in df_filtrado.columns:
        fig, ax = plt.subplots()
        df_filtrado["Tamaño de grano"].value_counts().plot(kind='bar', ax=ax)
        ax.set_xlabel("Tamaño de grano")
        ax.set_ylabel("Frecuencia")
        st.pyplot(fig)
    else:
        st.write("No se encontró la columna 'Tamaño de grano' para graficar.")

