import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from detect_image import mask_image
from detect import *


def detectOnWebcam():
    """
    Detect if people wear masks on the webcam video stream
    :return:
    """
    frameST = st.empty()
    print("* Start video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)  # Warm up the video

    faceNet, maskNet = loadNetworks()

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        locs, preds = predictMask(frame, faceNet, maskNet)

        frame = maskImage(frame, locs, preds)
        cv2.imshow('Mask Detection', frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        frameST.image(frame, channels="BGR")

    cv2.destroyAllWindows()
    vs.stop()


cv2.ocl.setUseOpenCL(False)

st.title("Face Mask Detection")
activities = ["Image", "Webcam"]
st.set_option('deprecation.showfileUploaderEncoding', False)
choice = st.sidebar.selectbox("The Input Format:", activities)

if choice == "Image":
    st.subheader("Mask Detection on Image")
    image_file = st.file_uploader("Upload Image", type=['jpg'])
    if image_file is not None:
        img = Image.open(image_file)
        img.save('./images/out.jpg')
        st.image(image_file, use_column_width=True)
        if st.button('Process'):
            loading_img_path = Image.open('./asset/loading.gif')
            loading_img = st.markdown("![Loading...](https://i.pinimg.com/originals/78/e8/26/78e826ca1b9351214dfdd5e47f7e2024.gif)")
            new_image = mask_image("./images/out.jpg")
            new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
            st.image(new_image, use_column_width=True)
            loading_img.empty()

if choice == "Webcam":
    detectOnWebcam()
