import pandas as pd
import streamlit as st

from joblib import load

best_model_RF = load('../models/best_model.joblib')
df = load('../data/processed/datos.df')

st.title('Predicción de Supervivencia en el Titanic')

passenger_id = st.number_input('PassengerId:', min_value=df.PassengerId.max()+1, step=1)
pclass = st.selectbox('Pclass:', sorted(df.Pclass.unique()))
name = st.text_input('Name:')
sex = st.selectbox('Sex:', df.Sex.unique())
age = st.number_input('Age:', min_value=0, max_value=100, step=1)
sibsp = st.number_input('SibSp:', min_value=0, max_value=15, step=1)
parch = st.number_input('Parch:', min_value=0, max_value=15, step=1)
ticket = st.text_input('Ticket:')
fare = st.slider('Fare:', min_value=0.0, max_value=800.0, step=1.0)
cabin = st.text_input('Cabin:')
embarked = st.selectbox('Embarked:', df.Embarked.dropna().unique())

if st.button('Enviar'):
    input_data = pd.DataFrame([[passenger_id, pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked]],
                               columns=['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'])

    predict = best_model_RF.predict(input_data)[0]

    if predict == 1:
        st.markdown(
            """
            <div style="background-image: url('https://cdn.aarp.net/content/aarpe/es/home/entretenimiento/cine-y-television/info-2022/anecdotas-fotos-videos-pelicula-titanic/_jcr_content/root/container_main/container_body_main/container_body2/container_body_cf/body_two_cf_one/par14/articlecontentfragme/cfimage.coreimg.50.932.jpeg/content/dam/aarp/entertainment/movies-for-grownups/2022/12/1140-titanic-set-esp.jpg'); height: 100vh; background-size: cover;">
                <h1 style="color: white;">¡Serías un Sobreviviente!</h1>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="background-image: url('https://media.vandalsports.com/i/640x360/7-2023/20237516291_1.jpg.webp'); height: 100vh; background-size: cover;">
                <h1 style="color: white;">No serías un Sobreviviente.</h1>
            </div>
            """,
            unsafe_allow_html=True
        )