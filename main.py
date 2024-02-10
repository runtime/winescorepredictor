import streamlit as st
import tensorflow as tf
import pandas as pd
from pathlib import Path
import pickle


if __name__ == "__main__":

    st.title("Wine Predictor")

    st.divider()

    file_path = Path('models/wine_quality.h5')

    nn_imported = tf.keras.models.load_model(file_path)
    print(nn_imported)

    # Evaluate the model using the test data
    #model_loss, model_accuracy = nn_imported.evaluate(X_test_scaled, y_test, verbose=2)

    # Display evaluation results
    #print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

    #import scaler here
    X_scaler = pickle.load(open('scalers/model_scaler.sav', 'rb'))


    with st.sidebar:
        fixed_acidity = st.number_input('Fixed_Acidity', step=.1)
        volatile_acidity = st.number_input('volatile_acidity', step=.1)
        citric_acidity = st.number_input('citric_acidity', step=.1)
        residue_sugar = st.number_input('residue_sugar', step=.1)
        chlorides = st.number_input('chlorides', step=.1)
        free_sulfur_dioxide = st.number_input('free_sulfur_dioxide', step=.1)
        total_sulfur_dioxide = st.number_input('total_sulfur_dioxide', step=.1)
        density = st.number_input('density', step=.1)
        ph = st.number_input('ph', step=.1)
        sulphates = st.number_input('sulphates', step=.1)
        alcohol = st.number_input('alcohol', step=.1)

        #print(fixed_acidity, volatile_acidity, citric_acidity, residue_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,ph, sulphates)

        characteristics = [[fixed_acidity,
                           volatile_acidity,
                           citric_acidity,
                           residue_sugar,
                           chlorides,
                           free_sulfur_dioxide,
                           total_sulfur_dioxide,
                           density,
                           ph,
                           sulphates,
                           alcohol]]

charcteristics_scaled = X_scaler.transform(characteristics)

if st.button("Predict", type="primary"):
    #prediction = nm_imported.predict(characteristics_scaled).round().astype("int32")
    prediction = nn_imported.predict(charcteristics_scaled).round().astype("int32")
    st.markdown(f"the predicted quality of the wine is {prediction[0][0]}.")

