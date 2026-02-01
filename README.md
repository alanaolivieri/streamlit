# Streamlit
Pequeño front con Streamlit de python para poner en ejecución un modelo de ML

Streamlit es un framework de código abierto para crear aplicaciones web interactivas con Python. Está diseñado para simplificar el proceso de convertir los scripts de análisis de datos en aplicaciones web interactivas de manera rápida y sencilla. Algunas características clave de Streamlit incluyen:

- Facilidad de Uso: Streamlit está diseñado para ser fácil de aprender y usar. Con unas pocas líneas de código, puedes convertir tus análisis de datos en aplicaciones web.

- Interactividad sin Esfuerzo: Permite a los usuarios interactuar con gráficos, tablas y otros elementos de manera intuitiva, sin necesidad de conocimientos avanzados de desarrollo web.

- Recarga Automática: Los cambios en el código se reflejan automáticamente en la aplicación, lo que facilita el proceso de desarrollo y prueba.

- Soporte para Gráficos Interactivos: Puedes integrar fácilmente gráficos interactivos generados con bibliotecas populares como Plotly y Matplotlib.

- Personalización: Aunque es fácil para principiantes, también permite un alto grado de personalización para usuarios más avanzados.

Instalación: pip install streamlit

Para la creación del Modelo se emplea Pipeline, es una secuencia de procesos que se aplican de manera encadenada para realizar tareas específicas, como preprocesamiento de datos, reducción de dimensiones y modelado. Estos pipelines son particularmente útiles para automatizar y estandarizar flujos de trabajo complejos, garantizando una ejecución eficiente y reproducible.

También nos permite que, cuando llevemos nuestro modelo a producción, aseguremos el tratamiento de los valores perdidos y el preprocesado de cada columna de la misma manera que se aplicó para los datos de entrenamiento.

Ejecución: 
- Para poner nuestro modelo en producción creamos una simple aplicación que permite la entrada de la información por parte del usuario y esta le responde si sobrevive o no al titanic. Ejecutamos en el terminal desde la carpeta scr: streamlit run app.py 
- Hacemos un simple dashboard para mostrar las visualizaciones que hemos realizado en nuestro notebook. Tenemos gráficos con plotly express, matplotlib, seaborn y matriz de confusion. Ejecutamos en el terminal desde la carpeta scr: streamlit run viz.py
