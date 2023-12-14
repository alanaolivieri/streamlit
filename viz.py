import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from joblib import load

# Cargamos el df
df = load('datos.df')
df_y = load('df_y.df')
df_metrics = load('df_metrics.df')

# Configuración de Streamlit
st.title('Visualización de Datos del Titanic')

# Sidebar para la selección de clases para las 3 primeras gráficas
selected_classes = st.multiselect('Selecciona Clases:', sorted(df['Pclass'].unique()), default=[1, 2, 3])

# Graficas tal cual nuestro notebook

### Graficos con pltly express

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

## Gráfico de Barras con la Media del Precio por Clase

# Calcular la media del Costo por cada combinación de Clase y Sobreviviente
df_media = df.groupby(['Pclass', 'Survived']).mean().reset_index()

fig_media = px.bar(df_media, x='Pclass', y='Fare', color='Survived', barmode="group", 
                title='Gráfico de Barras con la Media del Precio por Clase')
fig_media.update_xaxes(tickmode='linear', tick0=1, dtick=1)

st.plotly_chart(fig_media)


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

### Gráficos con sns

## Grafico de barras con métricas de los modelos con parámetros por defecto
st.header('Score de los modelos')
plt.figure(figsize=(8, 6)) # para mostrar la figura desde plt
sns.barplot(data=df_metrics, x = 'Modelo', y = 'Score')

st.pyplot(plt.gcf())

### Gráficos con matplotlib

## Grafico kde de las y_pred de los modelos frente al y_test
st.header('KDE para salidas')
fig_kde, ax = plt.subplots(figsize=(16, 10))
df_y.plot.kde(ax=ax)
plt.xlabel('Valor')
plt.ylabel('Densidad')

st.pyplot(fig_kde)

### Otros Graficos

## Matriz de confusión
st.header('Matriz de confusión: mejor modelo')
cm = confusion_matrix(df_y['y_test'], df_y['y_pred_best'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

fig, ax = plt.subplots()
disp.plot(ax=ax)  # Usar el mismo eje para ambos para que estén en la misma figura
st.pyplot(fig)

