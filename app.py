import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.colors as pc
from scipy.stats import linregress
from collections import defaultdict

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

    st.write("Datos cargados:")
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

    # Gráficos básicos
    st.header("Gráficos básicos")
    columnas = list(df_filtrado.columns)
    tipo_grafico = st.selectbox("Selecciona tipo de gráfico", ["Barra", "Pastel", "Heatmap (2 columnas)"])

    if tipo_grafico in ["Barra", "Pastel"]:
        columna = st.selectbox("Columna para graficar", columnas)
        data_graf = df_filtrado[columna].value_counts()

        fig, ax = plt.subplots(figsize=(8, 5))
        if tipo_grafico == "Barra":
            data_graf.plot(kind="bar", ax=ax)
        else:
            data_graf.plot(kind="pie", autopct="%1.1f%%", ax=ax)
        ax.set_title(f"{tipo_grafico} de {columna}")
        plt.figtext(0.5, -0.1, "Fuente: Cutipa, C. Jaramillo, A. Quenaya, F. Amaro, M.", ha="center", fontsize=9, style="italic")
        st.pyplot(fig)

    elif tipo_grafico == "Heatmap (2 columnas)":
        col1 = st.selectbox("Columna 1", columnas, key="col1")
        col2 = st.selectbox("Columna 2", columnas, key="col2")
        tabla = pd.crosstab(df_filtrado[col1], df_filtrado[col2])
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(tabla, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
        ax.set_title(f"Heatmap entre {col1} y {col2}")
        plt.figtext(0.5, -0.1, "Fuente: Cutipa, C. Jaramillo, A. Quenaya, F. Amaro, M.", ha="center", fontsize=9, style="italic")
        st.pyplot(fig)

    # Gráfico Sankey mejorado
    st.header("Gráfico innovador: Diagrama Sankey Mejorado")
    columnas_cat = [col for col in df_filtrado.columns if df_filtrado[col].dtype == object]
    cols_sankey = st.multiselect("Selecciona columnas categóricas (2 o 3)", columnas_cat, default=columnas_cat[:3])

    if len(cols_sankey) >= 2:
        labels = []
        for col in cols_sankey:
            labels.extend(df_filtrado[col].unique())
        labels = list(pd.Series(labels).unique())
        label_to_idx = {label: i for i, label in enumerate(labels)}

        source = []
        target = []
        value = []
        label_links = []

        for i in range(len(cols_sankey) - 1):
            df_grouped = df_filtrado.groupby([cols_sankey[i], cols_sankey[i + 1]]).size().reset_index(name='count')
            for _, row in df_grouped.iterrows():
                source.append(label_to_idx[row[cols_sankey[i]]])
                target.append(label_to_idx[row[cols_sankey[i + 1]]])
                value.append(row['count'])
                label_links.append(f"{row[cols_sankey[i]]} → {row[cols_sankey[i + 1]]}: {row['count']} muestras")

        colores_disponibles = pc.qualitative.Plotly
        colores_links = []
        for i in range(len(source)):
            color_base = colores_disponibles[i % len(colores_disponibles)]
            r = int(color_base[1:3], 16)
            g = int(color_base[3:5], 16)
            b = int(color_base[5:7], 16)
            colores_links.append(f'rgba({r},{g},{b},0.6)')

        # Mejor posicionamiento de nodos
        num_cols = len(cols_sankey)
        x_spacing = 1.0 / (num_cols - 1)
        node_x = []
        node_y = []

        for label in labels:
            nivel = next((i for i, col in enumerate(cols_sankey) if label in df_filtrado[col].values), 0)
            node_x.append(nivel * x_spacing)

        from collections import defaultdict
        nivel_nodes = defaultdict(list)
        for i, x in enumerate(node_x):
            nivel_nodes[x].append(i)
        node_y = [0] * len(labels)
        for nivel, nodes in nivel_nodes.items():
            n = len(nodes)
            for idx, node in enumerate(sorted(nodes)):
                node_y[node] = 1 - idx / (n - 1) if n > 1 else 0.5

        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color="skyblue",
                x=node_x,
                y=node_y
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color=colores_links,
                label=label_links,
                hovertemplate='%{label}<extra></extra>'
            )
        )])

        fig.update_layout(
            title_text="Diagrama Sankey Mejorado: Relaciones y Frecuencias",
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
    

  st.title("Dashboard sedimentario con cuestionario visual y Sankey dinámico")

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

    if "paso" not in st.session_state:
        st.session_state.paso = 1
        st.session_state.tipo_estruc = None
        st.session_state.tipo_estrat = None
        st.session_state.tam_grano = None

    # Paso 1 - Tipo de estructura sedimentaria
    if st.session_state.paso == 1:
        st.header("Paso 1: Selecciona tipo de estructura sedimentaria")
        opciones_estructura = df['estructura_sedimentaria'].dropna().unique()
        columnas = st.columns(min(len(opciones_estructura), 4))
        for i, opcion in enumerate(opciones_estructura):
            col = columnas[i % 4]
            if col.button(opcion):
                st.session_state.tipo_estruc = opcion
                st.session_state.paso = 2
                st.experimental_rerun()

    # Paso 2 - Tipo de estratificación
    elif st.session_state.paso == 2:
        st.header("Paso 2: Selecciona tipo de estratificación")
        opciones_estrat = df['tipo_de_estratificacion'].dropna().unique()
        columnas = st.columns(min(len(opciones_estrat), 4))
        for i, opcion in enumerate(opciones_estrat):
            col = columnas[i % 4]
            if col.button(opcion):
                st.session_state.tipo_estrat = opcion
                st.session_state.paso = 3
                st.experimental_rerun()

    # Paso 3 - Tamaño de grano
    elif st.session_state.paso == 3:
        st.header("Paso 3: Selecciona tamaño de grano")
        opciones_tamano = df['tamaño_de_grano'].dropna().unique()
        columnas = st.columns(min(len(opciones_tamano), 4))
        for i, opcion in enumerate(opciones_tamano):
            col = columnas[i % 4]
            if col.button(opcion):
                st.session_state.tam_grano = opcion
                st.session_state.paso = 4
                st.experimental_rerun()

    # Paso 4 - Resultados y Sankey
    elif st.session_state.paso == 4:
        st.header("Resultados y gráfico Sankey")

        filtro = (
            (df['estructura_sedimentaria'] == st.session_state.tipo_estruc) &
            (df['tipo_de_estratificacion'] == st.session_state.tipo_estrat) &
            (df['tamaño_de_grano'] == st.session_state.tam_grano)
        )
        df_filtrado = df[filtro]

        st.markdown(f"### {len(df_filtrado)} muestra(s) encontrada(s)")

        if len(df_filtrado) == 0:
            st.warning("No se encontraron muestras con esas características.")
        else:
            st.dataframe(df_filtrado[['muestra', 'estructura_sedimentaria', 'tipo_de_estratificacion', 'tamaño_de_grano']])

        cols_sankey = ['estructura_sedimentaria', 'tipo_de_estratificacion', 'tamaño_de_grano', 'muestra']
        df_sankey = df_filtrado if len(df_filtrado) > 0 else df

        labels = []
        for col in cols_sankey:
            labels.extend(df_sankey[col].unique())
        labels = list(pd.Series(labels).unique())
        label_to_idx = {label: i for i, label in enumerate(labels)}

        source = []
        target = []
        value = []
        label_links = []

        for i in range(len(cols_sankey) - 1):
            df_grouped = df_sankey.groupby([cols_sankey[i], cols_sankey[i + 1]]).size().reset_index(name='count')
            for _, row in df_grouped.iterrows():
                source.append(label_to_idx[row[cols_sankey[i]]])
                target.append(label_to_idx[row[cols_sankey[i + 1]]])
                value.append(row['count'])
                label_links.append(f"{row[cols_sankey[i]]} → {row[cols_sankey[i + 1]]}: {row['count']} muestras")

        colores_disponibles = pc.qualitative.Plotly
        colores_links = []
        for i in range(len(source)):
            color_base = colores_disponibles[i % len(colores_disponibles)]
            r = int(color_base[1:3], 16)
            g = int(color_base[3:5], 16)
            b = int(color_base[5:7], 16)
            colores_links.append(f'rgba({r},{g},{b},0.6)')

        num_cols = len(cols_sankey)
        x_spacing = 1.0 / (num_cols - 1)
        node_x = []
        node_y = []

        for label in labels:
            nivel = next((i for i, col in enumerate(cols_sankey) if label in df_sankey[col].values), 0)
            node_x.append(nivel * x_spacing)

        nivel_nodes = defaultdict(list)
        for i, x in enumerate(node_x):
            nivel_nodes[x].append(i)
        node_y = [0] * len(labels)
        for nivel, nodes in nivel_nodes.items():
            n = len(nodes)
            for idx, node in enumerate(sorted(nodes)):
                node_y[node] = 1 - idx / (n - 1) if n > 1 else 0.5

        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color="skyblue",
                x=node_x,
                y=node_y
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color=colores_links,
                label=label_links,
                hovertemplate='%{label}<extra></extra>'
            )
        )])

        fig.update_layout(
            title_text="Diagrama Sankey Dinámico: Características y muestras sedimentarias",
            font_size=10,
            annotations=[dict(
                text="Fuente: Cutipa, C. Jaramillo, A. Quenaya, F. Amaro, M.",
                x=0.5, y=-0.1, showarrow=False,
                font=dict(size=10, style="italic")
            )]
        )

        st.plotly_chart(fig, use_container_width=True)

    # Análisis de correlación con regresión
    st.header("Análisis de correlación y regresión")
    columnas_num = df_filtrado.select_dtypes(include='number').columns.tolist()
    if len(columnas_num) >= 2:
        col_x = st.selectbox("Variable X", columnas_num)
        col_y = st.selectbox("Variable Y", columnas_num, index=1)
        slope, intercept, r_value, p_value, std_err = linregress(df_filtrado[col_x], df_filtrado[col_y])
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(df_filtrado[col_x], df_filtrado[col_y], alpha=0.7, label="Datos")
        ax.plot(df_filtrado[col_x], intercept + slope * df_filtrado[col_x], color="red", label="Regresión lineal")
        ax.legend()
        ax.set_title(f"Relación entre {col_x} y {col_y}")
        st.pyplot(fig)
        st.markdown(f"**Coeficiente de correlación de Pearson:** {r_value:.3f} (p-valor: {p_value:.3g})")
        st.info("El coeficiente de Pearson mide la relación lineal entre dos variables numéricas. Cerca de 1 o -1 indica relación fuerte; cerca de 0, débil.")

else:
    st.info("Sube un archivo Excel corregido para comenzar.")


