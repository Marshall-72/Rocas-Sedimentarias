import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.colors as pc
from scipy.stats import linregress
import numpy as np

# Encabezado con imagen
st.image("https://esge.unjbg.edu.pe/portal-web/ingenieria-geologica-geotecnia/section/c2909719-d8de-4e37-97d3-42137a7651cf.png", width=1000)
st.title("Análisis y visualización de estructuras sedimentarias")

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

    # -----------------------------------------
    # FILTROS LATERALES
    # -----------------------------------------
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

    # -----------------------------------------
    # GRAFICOS BASICOS
    # -----------------------------------------
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

    # -----------------------------------------
    # GRAFICO SANKEY
    # -----------------------------------------
    st.header("Gráfico innovador: Diagrama Sankey mejorado")
    columnas_cat = [col for col in df_filtrado.columns if df_filtrado[col].dtype == object]
    cols_sankey = st.multiselect("Selecciona columnas categóricas (2 o 3)", columnas_cat, default=columnas_cat[:3])

    if len(cols_sankey) >= 2:
        labels = list(pd.unique(df_filtrado[cols_sankey].values.ravel('K')))
        label_idx = {k: v for v, k in enumerate(labels)}

        source, target, value = [], [], []

        # Agrupar para enlaces
        for i in range(len(cols_sankey) - 1):
            df_grouped = df_filtrado.groupby([cols_sankey[i], cols_sankey[i+1]]).size().reset_index(name='count')
            for _, row in df_grouped.iterrows():
                source.append(label_idx[row[cols_sankey[i]]])
                target.append(label_idx[row[cols_sankey[i+1]]])
                value.append(row['count'])

        # Distribuir nodos por nivel y posición vertical equidistante
        n_niveles = len(cols_sankey)
        x_spacing = 1 / (n_niveles - 1)

        # Crear lista de nodos por nivel
        nivel_nodos = {nivel: [] for nivel in range(n_niveles)}
        for i, label in enumerate(labels):
            for nivel_i, col in enumerate(cols_sankey):
                if label in df_filtrado[col].values:
                    nivel_nodos[nivel_i].append(i)
                    break

        node_x = []
        node_y = []

        for nivel in range(n_niveles):
            n = len(nivel_nodos[nivel])
            xs = [nivel * x_spacing] * n
            if n == 1:
                ys = [0.5]
            else:
                ys = list(np.linspace(0, 1, n))
            node_x.extend(xs)
            node_y.extend(ys)

        # Colorear enlaces por nodo origen (source)
        colores_base = pc.qualitative.Plotly
        link_colors = []
        for s in source:
            c = colores_base[s % len(colores_base)]
            r = int(c[1:3], 16)
            g = int(c[3:5], 16)
            b = int(c[5:7], 16)
            link_colors.append(f'rgba({r},{g},{b},0.6)')

        fig = go.Figure(data=[go.Sankey(
            node=dict(
                label=labels,
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                color="skyblue",
                x=node_x,
                y=node_y
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color=link_colors
            )
        )])

        fig.update_layout(
            title_text="Diagrama Sankey con distribución optimizada",
            font_size=10,
            annotations=[dict(
                text="Fuente: Cutipa, C. Jaramillo, A. Quenaya, F. Amaro, M.",
                x=0.5,
                y=-0.1,
                showarrow=False
            )]
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Selecciona al menos dos columnas categóricas para mostrar el diagrama Sankey.")

    # -----------------------------------------
    # PREGUNTAS INTERPRETATIVAS (independientes)
    # -----------------------------------------
    st.markdown("---")
    st.subheader("🧠 Respuestas generadas por IA - Preguntas interpretativas")
    st.markdown("**Selecciona una pregunta para ver su interpretación generada automáticamente por IA.**")

    preguntas_respuestas_dict = {
        "1. Indique tres tipos de estructuras sedimentarias propias de un determinado ambiente de sedimentación.": 
        "Tres tipos comunes de estructuras sedimentarias son: \n\n"
        "- **Estratificación cruzada**, típica de ambientes fluviales o desérticos donde los sedimentos se depositan con ángulos inclinados por el movimiento del agua o viento.\n"
        "- **Laminación paralela**, frecuente en ambientes tranquilos como lagos o plataformas marinas, donde los sedimentos se acumulan de forma ordenada en capas delgadas.\n"
        "- **Ondas de corriente (ripples)**, que se forman por el flujo de agua en ambientes someros como playas, ríos o deltas.",

        "2. ¿Qué tipo de estructuras sedimentarias son indicativas de ambientes continentales eólicos?": 
        "Los ambientes eólicos continentales generan estructuras como:\n\n"
        "- **Estratificación cruzada de gran escala**, formada por la migración de dunas de arena movidas por el viento.\n"
        "- **Superficies de deflación**, áreas donde el viento ha removido los sedimentos finos dejando gravas o pavimentos desérticos.\n"
        "- **Ripples eólicos**, pequeñas ondulaciones en la superficie del sedimento causadas por el arrastre de partículas finas por el viento.",

        "3. ¿En qué tipo de ambientes las trazas fósiles pueden ser encontradas como galerías? Explique.": 
        "Las trazas fósiles en forma de galerías son comunes en ambientes marinos someros y costeros, como playas, deltas o plataformas continentales. En estos ambientes, organismos como gusanos, moluscos o crustáceos excavan túneles en sedimentos blandos, generando estructuras biogénicas que quedan preservadas al litificarse el sedimento. Estas trazas reflejan condiciones de buena oxigenación y actividad biológica en el pasado geológico.",

        "4. ¿En qué tipo de ambientes se puede dar un tipo de bioturbación intensa?": 
        "La bioturbación intensa se da en ambientes sedimentarios con alta disponibilidad de oxígeno y organismos bentónicos, como plataformas continentales, zonas intermareales y fondos marinos litorales. En estos lugares, los organismos remueven activamente el sedimento, borrando o alterando las estructuras originales y dejando trazas que pueden ser estudiadas como ichnofósiles. Esta actividad biológica suele ser indicativa de estabilidad ambiental y baja tasa de sedimentación.",

        "5. ¿Puede un ambiente con sedimentación rápida generar buen registro icnofósil?": 
        "No. En ambientes con sedimentación rápida, los organismos no disponen del tiempo suficiente para excavar, alimentarse o dejar trazas significativas antes de quedar sepultados. Como resultado, el registro icnofósil es escaso o nulo. Estos entornos suelen estar asociados a procesos de alta energía como flujos de detritos, turbiditas o inundaciones súbitas.",

        "6. ¿Qué indica alternancia de estratos bioturbados y no bioturbados?": 
        "Esta alternancia indica variabilidad ambiental en el tiempo. Los estratos bioturbados reflejan periodos de baja sedimentación, buena oxigenación y presencia de fauna activa. Los estratos no bioturbados sugieren eventos de sedimentación rápida, condiciones anóxicas o ausencia de vida bentónica. Esta secuencia puede interpretarse como producto de ciclos climáticos, estacionales o eventos hidrodinámicos recurrentes.",

        "7. ¿Qué indica una laminación paralela?": 
        "La laminación paralela indica un ambiente de baja energía y sedimentación continua, como lagos profundos, llanuras de inundación o plataformas marinas externas. Las láminas reflejan deposición pausada de partículas finas que no son perturbadas por organismos o corrientes intensas. Su presencia sugiere estabilidad ambiental y transporte suspendido de sedimentos por largos periodos.",

        "8. ¿Qué estructuras presentan los ríos trenzados?": 
        "Los ríos trenzados presentan estratificación cruzada de gran escala, barras de arena y gravas, canales múltiples y bancos migratorios. Estas estructuras reflejan alta energía, carga sedimentaria abundante y cambios frecuentes en la dirección del flujo. La sedimentación se da de forma rápida y caótica, y los depósitos resultantes son mal clasificados y lateralmente discontinuos.",

        "9. ¿Qué estructuras presentan los ríos meándricos?": 
        "Los ríos meándricos muestran laminación paralela, estratificación planar y ocasionalmente estructuras de corte y relleno en sus canales. Sus depósitos están bien organizados y estratificados, con gradación normal. Reflejan ambientes de baja energía y flujo constante, como planicies de inundación o meandros abandonados donde predominan sedimentos finos como limos y arcillas.",

        "10. ¿Qué estructuras genera una corriente de turbidez?": 
        "Las corrientes de turbidez generan **estratificación gradada**. Este tipo de depósito, típico de ambientes marinos profundos como taludes continentales, se caracteriza por la disposición de partículas desde gruesas en la base hasta finas en el tope, producto de la decantación progresiva del flujo cargado de sedimentos. Esta secuencia es conocida como una turbidita o secuencia de Bouma."
    }

    pregunta_seleccionada = st.selectbox("Selecciona una pregunta:", list(preguntas_respuestas_dict.keys()))

    if pregunta_seleccionada:
        st.markdown("**Respuesta:**")
        st.info(preguntas_respuestas_dict[pregunta_seleccionada])

else:
    st.info("Sube un archivo Excel corregido para comenzar.")
