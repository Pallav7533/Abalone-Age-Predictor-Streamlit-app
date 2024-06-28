import streamlit as st
import numpy as np
import pickle


def prediction_age(Sex, Length, Diameter, Height, Whole_weight, Shucked_weight, Viscera_weight, Shell_weight):
    features = np.array([[Sex, Length, Diameter, Height, Whole_weight, Shucked_weight, Viscera_weight, Shell_weight]])
    with open('model.pkl', 'rb') as model_file:
        dtr = pickle.load(model_file)
    pred = dtr.predict(features).reshape(1, -1)
    return pred[0]

def main():
    
    with open('model.pkl', 'rb') as model_file:
        dtr = pickle.load(model_file)

    
    st.markdown('<h1 style="color: red;"> Abalone Age Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: #3366ff; font-size: 14px; font-family: Arial, sans-serif;">By <a href="https://github.com/Pallav7533" target="_blank" style="text-decoration: none; color: #3366ff;">Pal</a></p>', unsafe_allow_html=True)

    st.subheader('Enter Abalone Features:')
    Sex = st.selectbox('Sex (1 for Male, 2 for Female, 3 for Infant)', [1, 2, 3])
    Length = st.number_input('Length')
    Diameter = st.number_input('Diameter')
    Height = st.number_input('Height')
    Whole_weight = st.number_input('Whole Weight')
    Shucked_weight = st.number_input('Shucked Weight')
    Viscera_weight = st.number_input('Viscera Weight')
    Shell_weight = st.number_input('Shell Weight')

    
    if st.button('Predict Age'):
        prediction = prediction_age(Sex, Length, Diameter, Height, Whole_weight, Shucked_weight, Viscera_weight, Shell_weight)
        st.write(f"Predicted age: {prediction}")

       
        if prediction < 10:
            st.write("Category: Infant")
        elif prediction < 18:
            st.write("Category: Teenager")
        else:
            st.write("Category: Adult")

    
    image_path = 'abalone.png'  
    st.image(image_path, caption='Abalone Image', use_column_width=True)

if __name__ == '__main__':
    main()