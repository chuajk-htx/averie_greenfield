import tempfile
import streamlit as st
from Preprocess_Segment_Crop import openfile, Segmenter
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import glob
from streamlit_image_zoom import image_zoom
import base64
import time

st.title("Contact Lens Detection")


from PIL import Image  

# ------ Upload file ----------
"""
Image uploader widget, accepts single or multiple files
"""

uploaded_file = st.file_uploader("Upload an image file", type=["jpg","jpeg","bmp","png","tiff"], label_visibility="hidden", accept_multiple_files=True)

if uploaded_file != None:

    # Store a list of images from file uploader
    ls_images = []
    ls_clickable_images = []

    # Create a temp file in specified dir 
    direct = '/Users/averie/Documents/Algos/GF_demoapp/temp/'

    for item in uploaded_file:

        # Clickable image fxn requires base64 string representation of binary file
        rawbytes = item.getbuffer() # .read() or .getbuffer() gives the raw bytes of the uploaded file (IOBytes objects), but calling .read() to read again, it will return empty. whereas .getbuffer() does not advance the pointer
        encoded = base64.b64encode(rawbytes).decode() # converts raw bytes into a Base64-encoded bytes object, decode to convert Base64 bytes object into a string (UTF-8 text)
        ls_clickable_images.append(f"data:image/jpeg;base64,{encoded}")

        # Store uploaded image into the temp file, because upload file contains IOByte str, doesnt contain the file path
#            path = os.path.join(temp_dir, item.name) 
        path = os.path.join(direct, item.name)
        # Write to path
        with open(path, "wb") as f: 
            f.write(rawbytes)

        # Append images paths to list for iteration later 
        ls_images.append(path)

    
    # ------ Traveller info ----------

sec1_col1, sec1_col2 = st.columns([5, 1]) # 2:1 ratio of space

with sec1_col2:
    # KIV - make it either automated after processing or ...?
    if st.button('Clear files'):
        for filename in os.listdir(direct):
            file_path = os.path.join(direct, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)


################# Background Processes #####################

import math

rownum = math.ceil(len(uploaded_file)/2) 
#st.write(rownum)

import iris
# Initialize IRISPipeline object
iris_pipeline = iris.IRISPipeline()
db = '/Users/averie/Databases/NDCLD10/ND-Contact-2010'
matcher = iris.HammingDistanceMatcher()
def load_iris_template(img_pixel):
    img = iris_pipeline(img_pixel, eye_side="left")
    iriscode = img['iris_template']
    return iriscode

def get_id(file):
    try:
        if 'd' in file.name:
            subj = (file.name).split('.')[0].split('d')[0]
        else:
            subj = file.name.split('.')[0]
    except Exception as e:
        if 'd' in file:
            subj = file.split('/')[-1].split('.')[0].split('d')[0]
        else:
            subj = file.split('/')[-1].split('.')[0]
    return subj

from Prediction import predict

preds_list = []
gradcam_imgs_imgpixel = []

# ------ Image Preprocessing (Segment, Crop) ----------

with st.spinner("Processing... please wait ⏳"):

    # Preprocess all files in list of uploaded images
    for i_path in ls_images:
        im = openfile(i_path)

        seg = Segmenter(im).process().segment().cropping()
        processed = Segmenter(im).process().cropping()
#        st.image(seg.cropped, caption="Cropped Segmented Image", width=200)

        # save copy of segmented image 
        head, tail = os.path.split(i_path)
        output_filename = tail.split('.')[0] + '_seg' + '.png'
        output_file = os.path.join(head, output_filename) 
        cv2.imwrite(output_file,seg.cropped)

        # save copy of processed image for gradcam vis 
        output_processedfilename = tail.split('.')[0] + '_pro' + '.png'
        output_processedfile = os.path.join(head, output_processedfilename) # save segmented copy of image
        cv2.imwrite(output_processedfile,processed.cropped)


# ------ Model Prediction ----------
        seg_ls_images = []
        for segimgs in glob.glob(os.path.join(direct, '*_seg.png')):
            seg_ls_images.append(segimgs)

        pro_ls_images = []
        for proimgs in glob.glob(os.path.join(direct, '*_pro.png')):
            pro_ls_images.append(proimgs)

        pred_all, topk, gradcam = predict(seg_ls_images, pro_ls_images)
#        st.write(pred_all)
        preds_list = topk
        gradcam_imgs_imgpixel = gradcam

    st.toast("Processing Completed")
    time.sleep(0.15)

############################# UI ##############################

results = []



for row in range(0, rownum):
    col1, col2 = st.columns([2, 2]) # 2:1 ratio of space

    for idx, (imgpath, imgcode) in enumerate(zip(ls_images, uploaded_file), start=1):

        # Results recording
        results_breakdown = {
            "Name" : None,
            "Quality" : None,
            "Matching" : None,
            "CL Detection" : None,
        }

        if idx % 2 != 0:
            with col1:
                # "Traveller ID"
                st.write('')
                subj1 = get_id(imgcode)
                st.write("#### Traveller ID: ", subj1)
                st.image(imgcode, width=200)
                results_breakdown["Name"] = subj1

        # ------ Check if traveller in DB and get matching score ----------
        # ------ quality score ----------

                match = 0
                for data in glob.glob(os.path.join(db, '*')):
                    if subj1 in data and match < 1:
                        
                        img_pixels_db = cv2.imread(data, cv2.IMREAD_GRAYSCALE)
                        img_pixels_probe = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)


                        # ------ matching score ----------
                        db_subjcode = load_iris_template(img_pixels_db)
                        subjcode = load_iris_template(img_pixels_probe)
                        HD_distance = matcher.run(subjcode, db_subjcode)
                        match += 1

                        if HD_distance < 0.37: # threshold between 0.34-0.39, smaller the distance better is the match
                            st.markdown(
                            f"Matching: <span style='background-color:#e0ffe8; color:#155724; padding:2px 6px; border-radius:4px;'>Passed ({HD_distance:.2f})</span>",
                            unsafe_allow_html=True)
                            results_breakdown["Matching"] = "Pass"

                        else:
                            st.markdown(
                            f"Matching: <span style='background-color:#f8d7da; color:#721c24; padding:2px 6px; border-radius:4px;'>Failed ({HD_distance:.2f})</span>",
                            unsafe_allow_html=True)
                            results_breakdown["Matching"] = "Fail"

                if match < 1:
                    st.write("New Traveller")

        # ------ Check for Contact Lens Wearing ----------

                img_preds = preds_list[idx-1] # get dictionary by index

                top = 0 # first hit 
                img_name = img_preds[top]["name"]
                img_pred = img_preds[top]["class"]
                
                if img_pred == 'Cosmetic':
                    st.markdown(
                        f"Lens Detected: <span style='background-color:#f8d7da; color:#721c24; padding:2px 6px; border-radius:4px;'>{img_pred}</span>",
                        unsafe_allow_html=True)
                    results_breakdown["CL Detection"] = "Cosmetic"
                elif img_pred == 'Clear':
                    st.markdown(
                        f"Lens Detected: <span style='background-color:#cce5ff; color:#004085; padding:2px 6px; border-radius:4px;'>{img_pred}</span>",
                        unsafe_allow_html=True)
                    results_breakdown["CL Detection"] = "Clear"
                elif img_pred == 'No':
                    st.markdown(
                        f"No Lens Detected <span style='background-color:#ffffff; color:#000000; padding:2px 6px; border-radius:4px;'></span>",
                        unsafe_allow_html=True)
                    results_breakdown["CL Detection"] = "No"
                
                results.append(results_breakdown)

#----------- right image ----------#

        if idx % 2 == 0:
            with col2:
                st.write('')
                subj2 = get_id(imgcode)
                if subj2 != subj1:
                    st.write("#### Traveller ID: ", subj2)
                else:
                    f = ""
                    st.write(" ",f)
                    st.write("")
                    st.write("")
                st.image(imgcode, width=200)
                results_breakdown["Name"] = subj2

        # ------ Check if traveller in DB and get matching score ----------
        # ------ quality score ----------

                match = 0
                for data in glob.glob(os.path.join(db, '*')):
                    if subj2 in data and match < 1:
                        #st.write(data)
                        img_pixels_db = cv2.imread(data, cv2.IMREAD_GRAYSCALE)
                        img_pixels_probe = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)


                        # ------ matching score ----------
                        db_subjcode = load_iris_template(img_pixels_db)
                        subjcode = load_iris_template(img_pixels_probe)
                        HD_distance = matcher.run(subjcode, db_subjcode)
                        match += 1

                        if HD_distance < 0.37: # threshold between 0.34-0.39, smaller the distance better is the match
                            st.markdown(
                            f"Matching: <span style='background-color:#e0ffe8; color:#155724; padding:2px 6px; border-radius:4px;'>Pass ({HD_distance:.2f})</span>",
                            unsafe_allow_html=True)
                            results_breakdown["Matching"] = "Pass"

                        else:
                            st.markdown(
                            f"Matching: <span style='background-color:#f8d7da; color:#721c24; padding:2px 6px; border-radius:4px;'>Failed ({HD_distance:.2f})</span>",
                            unsafe_allow_html=True)
                            results_breakdown["Matching"] = "Fail"

                if match < 1:
                    st.write("New Traveller")

        # ------ Check for Contact Lens Wearing ----------

                img_preds = preds_list[idx-1] # get dictionary by index

                top = 0 # first hit 
                img_name = img_preds[top]["name"]
                img_pred = img_preds[top]["class"]
                
                if img_pred == 'Cosmetic':
                    st.markdown(
                        f"Lens Detected: <span style='background-color:#f8d7da; color:#721c24; padding:2px 6px; border-radius:4px;'>{img_pred}</span>",
                        unsafe_allow_html=True)
                    results_breakdown["CL Detection"] = "Cosmetic"
                elif img_pred == 'Clear':
                    st.markdown(
                        f"Lens Detected: <span style='background-color:#cce5ff; color:#004085; padding:2px 6px; border-radius:4px;'>{img_pred}</span>",
                        unsafe_allow_html=True)
                    results_breakdown["CL Detection"] = "Clear"
                elif img_pred == 'No':
                    st.markdown(
                        f"No Lens Detected <span style='background-color:#ffffff; color:#000000; padding:2px 6px; border-radius:4px;'></span>",
                        unsafe_allow_html=True)
                    results_breakdown["CL Detection"] = "No"

                results.append(results_breakdown)

    if idx == len(uploaded_file):

        with st.expander("#### Analysis"):


            tab1, tab2 = st.tabs(["Left", "Right"])

            for analys_idx, (path, _) in enumerate(zip(ls_images, uploaded_file), start=1):
 

                if analys_idx % 2 != 0: # in "Left" tab
                    
                    with tab1:
                        on = st.toggle("Activate feature", key=f"toggle_left_{analys_idx}")
                        
                        left_col1, left_col2 = st.columns([2,2])

                        img = openfile(ls_images[analys_idx-1])

                        with left_col1:
                            image_zoom(img, size=320, keep_resolution=True)
                        with left_col2:
                            if on:
                                image_zoom(gradcam_imgs_imgpixel[analys_idx-1], size=250, keep_resolution=True)

                        img_preds = preds_list[analys_idx-1] # get dictionary by index
                        for rank, pred in enumerate(img_preds, start=1):
                            st.write(f"**Rank {rank}** → {pred['class']} ({pred['prob']:.2f})")                    


                if analys_idx % 2 == 0:

                    with tab2:
                        on = st.toggle("Activate feature", key=f"toggle_left_{analys_idx}")

                        right_col1, right_col2 = st.columns([2,2])

                        img = openfile(ls_images[analys_idx-1])

                        with right_col1:
                            image_zoom(img, size=330, keep_resolution=True)
                        with right_col2:
                            if on:
                                image_zoom(gradcam_imgs_imgpixel[analys_idx-1], size=250, keep_resolution=True)

                        img_preds = preds_list[analys_idx-1] # get dictionary by index
                        for rank, pred in enumerate(img_preds, start=1):
                            st.write(f"**Rank {rank}** → {pred['class']} ({pred['prob']:.2f})")        

### Clearance Scenarios & Further Actions

for result in results: # iterate dictionaries in dictionary
    if result["CL Detection"] == 'Cosmetic':
        st.toast("Cosmetic lens detected \nPrompting fingerprint capture...")
        st.subheader("Fingerprint Capture")
    elif result["CL Detection"] == 'Clear' and result["Matching"] == 'Fail':
        st.toast("Matching failed \nTransparent lens detected \nPrompting fingerprint capture...")
        st.subheader("Fingerprint Capture")
    elif result["CL Detection"] == 'No' and result["Matching"] == 'Fail':
        st.toast("Matching failed \nPrompting iris recapture...")
        st.subheader("Iris ReCapture")
    elif result["Matching"] == "Pass":
        traveller = result["Name"]
        st.toast(f"Traveller {traveller} Verified")
        st.subheader("")

