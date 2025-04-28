import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import streamlit as st
import cv2

model = tf.keras.models.load_model("./model",custom_objects={'KerasLayer': hub.KerasLayer})
st.title("MALWARE DETECTION - DEEP LEARNING ALGORITHM")
st.subheader("Malware Detection App by Desmond Kwaku Duah")
uploaded_file = st.file_uploader("Upload malware image here:", type="png")

labels = {0: 'Adialer.C - Dialer',
          1: 'Agent.FYI - Backdoor',
          2: 'Allaple.A - Worm',
          3: 'Allaple.L - Worm',
          4: 'Alueron.gen!J - Trojan',
          5: 'Autorun.K - Worm:AutoIT',
          6: 'C2Lop.P - Trojan',
          7: 'C2Lop.gen!g - Trojan',
          8: 'Dialplatform.B - Dialer',
          9: 'Dontovo.A - TDownloader',
          10: 'Fakerean - Rogue',
          11: 'Instantaccess - Dialer',
          12: 'Lolyda.AA 1 - PWS',
          13: 'Lolyda.AA 2- PWS',
          14: 'Lolyda.AA 3- PWS',
          15: 'Lolyda.AT - PWS',
          16: 'Malex.gen!J - Trojan',
          17: 'Obfuscator.AD - TDownloader',
          18: 'Rbot!gen - Backdoor',
          19: 'Skintrim.N - Trojan',
          20: 'Swizzor.gen!E - TDownloader',
          21: 'Swizzot.gen!I - TDownloader',
          22: 'VB.AT - Worm',
          23: 'Wintrim.BX - TDownloader',
          24: 'Yuner.A - Worm'}

if uploaded_file is not None:
    with st.spinner("Detecting..."):
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(opencv_image, (224, 224))

        st.image(opencv_image, channels="RGB")

        resized_image = tf.keras.applications.mobilenet_v2.preprocess_input(resized_image)
        img_reshaped = resized_image[np.newaxis, ...]

        predicted_value = model.predict(img_reshaped)
        label_key = np.argmax(predicted_value)
        st.title(f"Predicted Label: {labels[label_key]}")
