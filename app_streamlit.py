import streamlit as st
import numpy as np
import urllib.request
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import gdown
import os
import zipfile

# URL del modelo en Google Drive
zip_url = 'https://drive.google.com/file/d/1U96luzv8S4RLlUI6np_ZR7JgmM8QduA3/view?usp=sharing'
zip_filename = 'retinopathy_detection_finetunning.keras.zip'
model_filename = 'retinopathy_detection_finetunning.keras'

if not os.path.exists(zip_filename):
    with st.spinner('Descargando el archivo ZIP...'):
        gdown.download(zip_url, zip_filename)
    st.success('Archivo ZIP descargado con éxito!')

# Extraer el archivo .h5 del ZIP si no está en la carpeta
if not os.path.exists(model_filename):
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extract(model_filename)
    st.success(f'Archivo {model_filename} extraído del ZIP con éxito!')

# Título de la aplicación
st.title('Predicción de Imágenes con Modelo de Deep Learning')

# Verificación de carga de archivo
uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Cargar el modelo
        modelo = load_model(model_filename)

        # Mostrar la imagen subida
        st.image(uploaded_file, caption='Imagen de entrada', use_column_width=True)

        # Preprocesamiento de la imagen para hacer la predicción
        img = image.load_img(uploaded_file, target_size=(229, 229))  # Ajusta según las dimensiones de entrada de tu modelo
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.

        # Realizar la predicción
        prediction = modelo.predict(img_array)

        # Mostrar resultados
        st.write("Predicción:")
        st.write(prediction)

    except Exception as e:
        st.write("Error al cargar el modelo o procesar la imagen. Detalles del error:", e)
