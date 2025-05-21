import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.colors as pc
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

    st.write("Datos cargados:")
    st.dataframe(df)

    # Filtros laterales
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

    # Gráfico Sankey
    st.header("Gráfico innovador: Diagrama Sankey")
    columnas_cat = [col for col in df_filtrado.columns if df_filtrado[col].dtype == object]
    cols_sankey = st.multiselect("Selecciona columnas categóricas (2 o 3)", columnas_cat, default=columnas_cat[:3])

    if len(cols_sankey) >= 2:
        labels = list(pd.unique(df_filtrado[cols_sankey].values.ravel('K')))
        label_idx = {k: v for v, k in enumerate(labels)}
        source, target, value = [], [], []
        for i in range(len(cols_sankey) - 1):
            df_grouped = df_filtrado.groupby([cols_sankey[i], cols_sankey[i+1]]).size().reset_index(name='count')
            for _, row in df_grouped.iterrows():
                source.append(label_idx[row[cols_sankey[i]]])
                target.append(label_idx[row[cols_sankey[i+1]]])
                value.append(row['count'])
        colores = pc.qualitative.Plotly
        link_colors = [f'rgba{tuple(int(colores[i % len(colores)][j:j+2], 16) for j in (1,3,5)) + (0.6,)}' for i in range(len(source))]
        fig = go.Figure(data=[go.Sankey(
            node=dict(label=labels, pad=15, thickness=20, color="skyblue"),
            link=dict(source=source, target=target, value=value, color=link_colors)
        )])
        fig.update_layout(title_text="Diagrama Sankey", font_size=10,
                          annotations=[dict(text="Fuente: Cutipa, C. Jaramillo, A. Quenaya, F. Amaro, M.", x=0.5, y=-0.1, showarrow=False)])
        st.plotly_chart(fig, use_container_width=True)

    # Correlación con regresión
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

    # Preguntas interpretativas
  # --- Sección de respuestas interpretativas generadas por IA ---
try:
    st.markdown("---")
    st.subheader("🧠 Respuestas generadas por IA - Preguntas interpretativas")
    st.markdown("**Nota:** Las siguientes respuestas han sido generadas automáticamente mediante inteligencia artificial con fines educativos.")

    preguntas_respuestas = [
        ("1. Indique tres tipos de estructuras sedimentarias propias de un determinado ambiente de sedimentación.",
         "Estratificación cruzada, laminación paralela y ondas de corriente."),
        ("2. ¿Qué tipo de estructuras sedimentarias son indicativas de ambientes continentales eólicos?",
         "Estratificación cruzada eólica, superficies de deflación y ripples formados por el viento."),
        ("3. ¿En qué tipo de ambientes las trazas fósiles pueden ser encontradas como galerías?",
         "En ambientes marinos costeros y plataformas someras con sedimentos blandos, oxigenados, donde organismos excavan galerías."),
        ("4. ¿En qué tipo de ambientes se puede dar un tipo de bioturbación intensa?",
         "En ambientes marinos someros con buena oxigenación y abundante fauna bentónica."),
        ("5. ¿Puede un ambiente con sedimentación rápida generar buen registro icnofósil?",
         "No, porque los organismos no tienen tiempo suficiente para excavar o dejar trazas antes del enterramiento."),
        ("6. ¿Qué indica alternancia de estratos bioturbados y no bioturbados?",
         "Indica variaciones ambientales: periodos de alta y baja actividad biológica o cambios en la tasa de sedimentación."),
        ("7. ¿Qué indica una laminación paralela?",
         "Ambientes de baja energía como lagos o mares profundos, con sedimentación lenta y ordenada."),
        ("8. ¿Qué estructuras presentan los ríos trenzados?",
         "Estratificación cruzada de gran escala, barras arenosas y sedimentos gruesos en múltiples canales."),
        ("9. ¿Qué estructuras presentan los ríos meándricos?",
         "Laminación paralela y planar en depósitos de llanuras de inundación con sedimentos finos."),
        ("10. ¿Qué estructuras genera una corriente de turbidez?",
         "Estratificación gradada, donde los sedimentos se ordenan por tamaño desde grueso (base) a fino (tope).")
    ]

    for pregunta, respuesta in preguntas_respuestas:
        st.markdown(f"**{pregunta}**")
        st.markdown(f"> {respuesta}")
        st.markdown("---")

except Exception as e:
    st.error(f"No se pudo mostrar la sección de preguntas interpretativas: {e}")

else:
    st.info("Sube un archivo Excel corregido para comenzar.")


