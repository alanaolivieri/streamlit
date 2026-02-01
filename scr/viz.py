import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from joblib import load

df = load('../data/processed/datos.df')
df_y = load('../data/processed/df_y.df')
df_metrics = load('../data/processed/df_metrics.df')

st.title('Visualización de Datos del Titanic')

selected_classes = st.multiselect('Selecciona Clases:', sorted(df['Pclass'].unique()), default=[1, 2, 3])

### Graficos con plotly express

fig_scatter = px.scatter(df[df['Pclass'].isin(selected_classes)], x='Fare', y='Age', color='Pclass',
                         title='Diagrama de dispersión por Clase')

fig_scatter.update_layout(coloraxis_colorbar=dict(
    title='Clase',
    tickvals=[1, 2, 3],
    ticktext=['Clase 1', 'Clase 2', 'Clase 3']
))

st.plotly_chart(fig_scatter)

fig_histogram = px.histogram(df[df['Pclass'].isin(selected_classes)], x='Age', color='Pclass',
                              title='Histograma de Edades por Clase')

st.plotly_chart(fig_histogram)

fig_box = px.box(df, x="Pclass", y="Fare", color="Survived", title='Clase y costo del pasaje según supervivencia')

st.plotly_chart(fig_box)

fig_bar = px.bar(df.groupby(['Pclass', 'Survived'])['Fare'].mean().reset_index(), x='Pclass', y='Fare', color='Survived', title='Gráfico de Barras con la Media del Precio por Clase')
fig_bar.update_xaxes(dtick=1)

st.plotly_chart(fig_bar)


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

st.header('Score de los modelos')
plt.figure(figsize=(8, 6)) # para mostrar la figura desde plt
sns.barplot(data=df_metrics, x = 'Modelo', y = 'Score')

st.pyplot(plt.gcf())

### Gráficos con matplotlib

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
disp.plot(ax=ax)
st.pyplot(fig)

