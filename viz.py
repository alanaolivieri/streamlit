import streamlit as st
import pandas as pd
import plotly.express as px

from joblib import load

# Cargamos el df
df = load('datos.df')

# Configuración de Streamlit
st.title('Visualización de Datos del Titanic')

# Cambiar el fondo a negro con un contenedor HTML---- esta fallando ojooooooooo
st.markdown(
    """
    <style>
        body {
            background-color: black;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar para la selección de clases para las 3 primeras gráficas
selected_classes = st.multiselect('Selecciona Clases:', sorted(df['Pclass'].unique()), default=[1, 2, 3])

# Graficas tal cual nuestro notebook
## Diagrama de dispersion

fig_scatter = px.scatter(df[df['Pclass'].isin(selected_classes)], x='Fare', y='Age', color='Pclass',
                         title='Diagrama de dispersión por Clase')

fig_scatter.update_layout(coloraxis_colorbar=dict(
    title='Clase',
    tickvals=sorted(selected_classes),
    ticktext=[f'Clase {cls}' for cls in sorted(selected_classes)]
))

st.plotly_chart(fig_scatter)

## Histograma de Edades por Clase

fig_histogram = px.histogram(df[df['Pclass'].isin(selected_classes)], x='Age', color='Pclass',
                              title='Histograma de Edades por Clase')

st.plotly_chart(fig_histogram)

## Gráfico de Barras Apiladas por Supervivencia y Clase

fig_bar = px.bar(df[df['Pclass'].isin(selected_classes)], x='Pclass', y='Fare', color='Survived',
                 title='Gráfico de Barras Apiladas por Supervivencia y Clase')

fig_bar.update_layout(coloraxis_colorbar=dict(
    tickvals=[0, 1],
    ticktext=[0, 1]
))

st.plotly_chart(fig_bar)

## Gráfico de dispersión animado para Clase seleccionada

# Para seleccionar 1 sola clase
selected_class = st.radio('Selecciona una Clase:', sorted(df['Pclass'].unique()), index=0)

fig_scatter_animation = px.scatter(df[(df['Pclass'] == selected_class) & (df['Embarked'] == 'S')],
                                   x='Age', y='Fare', animation_frame='Fare',
                                   title=f'Relación entre Edad, Tarifa y Clase {selected_class}')

fig_scatter_animation.update_layout(xaxis_title='Edad', yaxis_title='Tarifa')

fig_scatter_animation.update_layout(xaxis=dict(range=[0, 80]),
                                     yaxis=dict(range=[0, df.Fare[(df['Pclass'] == selected_class) & (df['Embarked'] == 'S')].max() + 10]))

fig_scatter_animation.update_traces(marker=dict(size=18, opacity=0.5))

st.plotly_chart(fig_scatter_animation)
