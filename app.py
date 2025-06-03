import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.colors as pc
from collections import defaultdict
from scipy.stats import linregress

st.title("Dashboard sedimentario con cuestionario, gráficos, Sankey y preguntas interpretativas")

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

    # --- Cuestionario visual con botones ---
    if "paso" not in st.session_state:
        st.session_state.paso = 1
        st.session_state.tipo_estruc = None
        st.session_state.tipo_estrat = None
        st.session_state.tam_grano = None

    if st.session_state.paso == 1:
        st.header("Paso 1: Selecciona tipo de estructura sedimentaria")
        opciones_estructura = df['estructura_sedimentaria'].dropna().unique()
        columnas = st.columns(min(len(opciones_estructura), 4))
        for i, opcion in enumerate(opciones_estructura):
            col = columnas[i % 4]
            if col.button(opcion):
                st.session_state.tipo_estruc = opcion
        if st.session_state.tipo_estruc:
            if st.button("Continuar al Paso 2"):
                st.session_state.paso = 2

    elif st.session_state.paso == 2:
        st.header("Paso 2: Selecciona tipo de estratificación")
        opciones_estrat = df['tipo_de_estratificacion'].dropna().unique()
        columnas = st.columns(min(len(opciones_estrat), 4))
        for i, opcion in enumerate(opciones_estrat):
            col = columnas[i % 4]
            if col.button(opcion):
                st.session_state.tipo_estrat = opcion
        if st.session_state.tipo_estrat:
            if st.button("Continuar al Paso 3"):
                st.session_state.paso = 3

    elif st.session_state.paso == 3:
        st.header("Paso 3: Selecciona tamaño de grano")
        opciones_tamano = df['tamaño_de_grano'].dropna().unique()
        columnas = st.columns(min(len(opciones_tamano), 4))
        for i, opcion in enumerate(opciones_tamano):
            col = columnas[i % 4]
            if col.button(opcion):
                st.session_state.tam_grano = opcion
        if st.session_state.tam_grano:
            if st.button("Mostrar Resultados y Gráficos"):
                st.session_state.paso = 4

    elif st.session_state.paso == 4:
        st.header("Resultados y visualizaciones")

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

        # --- Gráficos básicos ---
        st.subheader("Gráficos básicos")

        tipo_grafico = st.selectbox("Selecciona tipo de gráfico", ["Barra", "Pastel", "Heatmap (2 columnas)"])

        if tipo_grafico in ["Barra", "Pastel"]:
            columnas = list(df_filtrado.columns)
            columna = st.selectbox("Columna para graficar", columnas, key="grafico_basico_col")
            if columna:
                data_graf = df_filtrado[columna].value_counts()
                fig, ax = plt.subplots(figsize=(8, 5))
                if tipo_grafico == "Barra":
                    data_graf.plot(kind="bar", ax=ax)
                else:
                    data_graf.plot(kind="pie", autopct="%1.1f%%", ax=ax)
                ax.set_title(f"{tipo_grafico} de {columna}")
                plt.figtext(0.5, -0.1, "Fuente: Cutipa, C. Jaramillo, A. Quenaya, F. Amaro, M.", ha="center", fontsize=9, style="italic")
                st.pyplot(fig)
            else:
                st.warning("Selecciona una columna para graficar.")

        elif tipo_grafico == "Heatmap (2 columnas)":
            col1 = st.selectbox("Columna 1", df_filtrado.columns, key="heatmap_col1")
            col2 = st.selectbox("Columna 2", df_filtrado.columns, key="heatmap_col2")
            if col1 and col2 and col1 != col2:
                tabla = pd.crosstab(df_filtrado[col1], df_filtrado[col2])
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(tabla, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
                ax.set_title(f"Heatmap entre {col1} y {col2}")
                plt.figtext(0.5, -0.1, "Fuente: Cutipa, C. Jaramillo, A. Quenaya, F. Amaro, M.", ha="center", fontsize=9, style="italic")
                st.pyplot(fig)
            else:
                st.warning("Selecciona dos columnas diferentes para el heatmap.")

        # --- Sankey dinámico ---
        st.subheader("Diagrama Sankey dinámico")
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

        # --- Correlación y regresión ---
        st.subheader("Análisis de correlación y regresión")

        columnas_num = df_filtrado.select_dtypes(include='number').columns.tolist()
        if len(columnas_num) >= 2:
            col_x = st.selectbox("Variable X", columnas_num, key="corr_x")
            col_y = st.selectbox("Variable Y", columnas_num, key="corr_y")
            if col_x != col_y:
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
                st.warning("Selecciona dos variables diferentes para el análisis de correlación.")

    # --- Preguntas interpretativas ---
    st.header("Preguntas interpretativas - respuestas generadas por IA")
    preguntas_respuestas_dict = {
        "1. Indique tres tipos de estructuras sedimentarias propias de un determinado ambiente de sedimentación.":
            "Tres tipos comunes de estructuras sedimentarias son: estratificación cruzada, laminación paralela y ondas de corriente.",
        "2. ¿Qué tipo de estructuras sedimentarias son indicativas de ambientes continentales eólicos?":
            "En ambientes eólicos continentales son comunes la estratificación cruzada de gran escala, superficies de deflación y ripples eólicos.",
        "3. ¿En qué tipo de ambientes las trazas fósiles pueden ser encontradas como galerías?":
            "En ambientes marinos someros y costeros con sedimentos blandos y oxigenados, organismos excavan galerías preservadas como trazas fósiles.",
        "4. ¿En qué tipo de ambientes se puede dar un tipo de bioturbación intensa?":
            "En ambientes marinos someros bien oxigenados con abundante fauna bentónica se produce bioturbación intensa.",
        "5. ¿Puede un ambiente con sedimentación rápida generar buen registro icnofósil?":
            "No, porque la sedimentación rápida impide que los organismos formen trazas antes de quedar sepultados.",
        "6. ¿Qué indica alternancia de estratos bioturbados y no bioturbados?":
            "Indica variabilidad ambiental con periodos de actividad biológica alternados con sedimentación rápida o anóxica.",
        "7. ¿Qué indica una laminación paralela?":
            "Refleja ambientes de baja energía con sedimentación continua y estable, como lagos o plataformas marinas profundas.",
        "8. ¿Qué estructuras presentan los ríos trenzados?":
            "Los ríos trenzados presentan estratificación cruzada, barras arenosas y canales múltiples.",
        "9. ¿Qué estructuras presentan los ríos meándricos?":
            "Los ríos meándricos muestran laminación paralela y estratificación planar en llanuras de inundación.",
        "10. ¿Qué estructuras genera una corriente de turbidez?":
            "Genera estratificación gradada típica de flujos densos con partículas que se depositan por tamaño."
    }
    pregunta_seleccionada = st.selectbox("Selecciona una pregunta para ver la respuesta generada por IA:", list(preguntas_respuestas_dict.keys()))
    if pregunta_seleccionada:
        st.markdown(f"**Respuesta:**")
        st.info(preguntas_respuestas_dict[pregunta_seleccionada])

else:
    st.info("Sube un archivo Excel corregido para comenzar.")

