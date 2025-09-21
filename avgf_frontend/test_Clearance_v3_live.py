
import tempfile
import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os, sys
import glob
from streamlit_image_zoom import image_zoom
import base64
import time
import PIL
import glob


parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path(parent_dir)
from contact_lens_predictor.Preprocess_Segment_Crop import openfile, Segmenter

def main():
    st.set_page_config(page_title="Live Analysis", layout="wide")
    st.title("Live Analysis")

    ls_images = glob.glob(os.path.join(parent_dir, "received_file_images", "*.jpg"),reversed=False)[:2]
    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_filepath = temp_file.name

        # Display the uploaded image
        image = PIL.Image.open(temp_filepath)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Process the image using Segmenter
        segmenter = Segmenter()
        processed_image = segmenter.process_image(temp_filepath)

        # Display the processed image
        st.image(processed_image, caption='Processed Image', use_column_width=True)

        # Clean up the temporary file
        os.remove(temp_filepath)
    