import tempfile
import streamlit as st
from Preprocess_Segment_Crop import openfile, Segmenter
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from streamlit_image_zoom import image_zoom
import glob

#st.title("Contact Lens Detection")

import streamlit as st

pages = {
    "Demonstration Use Cases": [
#        st.Page("Enrolment.py", title="Enrolment"),
        st.Page("Clearance_v3.py", title="Clearance"),
    ],
}

pg = st.navigation(pages, position="sidebar", expanded=True)
pg.run()

