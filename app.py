import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.colors as pc
import scipy.stats as stats
from scipy.stats import linregress

st.title("Dashboard completo: Análisis y visualización de estructuras sedimentarias")

def agregar_columnas_numericas(df):
    mapa_tamano = {
        "muy fino": 10,
        "fino": 50,
        "medio": 200,
        "grueso": 600,
        "muy grueso": 1000
    }
    df['granulometria_um'] = df['tamaño_de_grano'].map(mapa_tamano).fillna(0)

    mapa_estrat = {
        "Estratificación plana": 1,
        "Estratificación cruzada": 2,
        "Estratificación interna": 3
    }
    df['complejidad_estratificacion'] = df['tipo_de_estratificacion'].map(mapa_estrat).fillna(0)

    return df

uploaded_file = st.file_uploader("Sube tu archivo Excel corregido", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df = agregar_columnas_numericas(df)

    st.write("Datos cargados (con columnas numéricas añadidas):")
    st.dataframe(df)

    st.sidebar.header("Filtros")
    filtros = {}
    for col in df.columns:
        if df[col].dtype == object:
            opciones = df[col].dropna().unique()
            seleccion = st.sidebar.multiselect(f"Filtrar por {col}", opciones, default=opciones)
            filtros[col] = seleccion
        else:
            minimo = float(df[col].min())
            maximo = float(df[col].max())
            rango = st.sidebar.slider(f"Rango para {col}", minimo, maximo, (minimo, maximo))
            filtros[col] = rango

    df_filtrado = df.copy()
    for col, val in filtros.items():
        if isinstance(val, list):
            df_filtrado = df_filtrado[df_filtrado[col].isin(val)]
        else:
            df_filtrado = df_filtrado[(df_filtrado[col] >= val[0]) & (df_filtrado[col] <= val[1])]

    st.write(f"Datos filtrados: {len(df_filtrado)} registros")
    st.dataframe(df_filtrado)

    st.header("Gráficos básicos")
    columnas = list(df_filtrado.columns)
    tipo_grafico = st.selectbox("Selecciona tipo de gráfico básico", ["Barra", "Pastel", "Heatmap (2 columnas)"])

    if tipo_grafico in ["Barra", "Pastel"]:
        columna = st.selectbox("Selecciona la columna para graficar", columnas)
        data_graf = df_filtrado[columna].value_counts()

        fig, ax = plt.subplots(figsize=(8,5))
        if tipo_grafico == "Barra":
            data_graf.plot(kind="bar", ax=ax)
            ax.set_ylabel("Frecuencia")
            ax.set_xlabel(columna)
            ax.set_title(f"Gráfico de barras - {columna}")
        else:
            data_graf.plot(kind="pie", autopct="%1.1f%%", ax=ax)
            ax.set_ylabel("")
            ax.set_title(f"Gráfico de pastel - {columna}")

        # Fuente abajo del gráfico
        plt.figtext(0.5, -0.1, "Fuente: Cutipa, C. Jaramillo, A. Quenaya, F. Amaro, M.", ha="center", fontsize=9, style="italic")

        st.pyplot(fig)
    else:
        col1 = st.selectbox("Selecciona columna 1 para heatmap", columnas, key="col1")
        col2 = st.selectbox("Selecciona columna 2 para heatmap", columnas, key="col2")
        tabla_cruzada = pd.crosstab(df_filtrado[col1], df_filtrado[col2])

        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(tabla_cruzada, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
        ax.set_title(f"Heatmap de frecuencias entre {col1} y {col2}")

        plt.figtext(0.5, -0.1, "Fuente: Cutipa, C. Jaramillo, A. Quenaya, F. Amaro, M.", ha="center", fontsize=9, style="italic")

        st.pyplot(fig)

    st.header("Gráfico innovador: Diagrama Sankey")
    columnas_categoricas = [col for col in df_filtrado.columns if df_filtrado[col].dtype == object]
    st.write("Selecciona columnas categóricas (2 o 3) para Sankey")
    cols_sankey = st.multiselect("Columnas para Sankey", columnas_categoricas, default=columnas_categoricas[:3])

    if len(cols_sankey) >= 2:
        labels = []
        for col in cols_sankey:
            labels.extend(df_filtrado[col].unique())
        labels = list(pd.Series(labels).unique())
        label_to_idx = {label: i for i, label in enumerate(labels)}

        source = []
        target = []
        value = []

        for i in range(len(cols_sankey) - 1):
            df_grouped = df_filtrado.groupby([cols_sankey[i], cols_sankey[i + 1]]).size().reset_index(name='count')
            for _, row in df_grouped.iterrows():
                source.append(label_to_idx[row[cols_sankey[i]]])
                target.append(label_to_idx[row[cols_sankey[i + 1]]])
                value.append(row['count'])

        num_links = len(source)
        colores_disponibles = pc.qualitative.Plotly
        colores_links = []
        for i in range(num_links):
            color_base = colores_disponibles[i % len(colores_disponibles)]
            r = int(color_base[1:3], 16)
            g = int(color_base[3:5], 16)
            b = int(color_base[5:7], 16)
            color_rgba = f'rgba({r},{g},{b},0.6)'
            colores_links.append(color_rgba)

        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color="skyblue"
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color=colores_links
            )
        )])

        fig.update_layout(
            title_text="Diagrama Sankey de estructuras sedimentarias",
            font_size=10,
            annotations=[dict(
                text="Fuente: Cutipa, C. Jaramillo, A. Quenaya, F. Amaro, M.",
                x=0.5, y=-0.1, showarrow=False,
                font=dict(size=10, style="italic")
            )]
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Selecciona al menos dos columnas categóricas para generar el Sankey.")

    st.header("Análisis de correlación")
    columnas_numericas = df_filtrado.select_dtypes(include=['number']).columns.tolist()

    if len(columnas_numericas) < 2:
        st.info("No hay suficientes columnas numéricas para analizar correlación.")
    else:
        col_x = st.selectbox("Variable X (numérica)", columnas_numericas, key="corr_x")
        col_y = st.selectbox("Variable Y (numérica)", columnas_numericas, index=1 if len(columnas_numericas) > 1 else 0, key="corr_y")

        slope, intercept, r_value, p_value, std_err = linregress(df_filtrado[col_x], df_filtrado[col_y])

        fig, ax = plt.subplots(figsize=(8,5))
        ax.scatter(df_filtrado[col_x], df_filtrado[col_y], alpha=0.7, label="Datos")
        ax.plot(df_filtrado[col_x], intercept + slope * df_filtrado[col_x], color="red", label="Regresión lineal")
        ax.set_xlabel(col_x)
        ax.set_ylabel(col_y)
        ax.set_title(f"Scatter plot y regresión lineal entre {col_x} y {col_y}")
        ax.legend()

        st.pyplot(fig)

        st.write(f"Coeficiente de correlación de Pearson: {r_value:.3f} (p-valor = {p_value:.3g})")

        descripcion = """
        El coeficiente de correlación de Pearson mide la relación lineal entre dos variables numéricas.
        Valores cercanos a 1 o -1 indican una relación fuerte positiva o negativa, respectivamente.
        Valores cercanos a 0 indican poca o ninguna relación lineal.
        """
        st.info(descripcion)
else:
    st.info("Sube un archivo Excel corregido para comenzar.")


