import streamlit as st
import os
import requests
import base64
import io
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import random
import math
from roboflow import Roboflow
from collections import Counter

#TODO figure out if workspace_id variable can be setup for having a session_state
project_url_od, private_api_key, uploaded_file_od, pill_input_array = ("", "", "", [])

if 'project_url_od' not in st.session_state:
    st.session_state['project_url_od'] = "https://app.roboflow.com/ms-cs-thesis/medication-identification-v2/2"
if 'confidence_threshold' not in st.session_state:
    st.session_state['confidence_threshold'] = "40"
if 'overlap_threshold' not in st.session_state:
    st.session_state['overlap_threshold'] = "30"
if 'include_bbox' not in st.session_state:
    st.session_state['include_bbox'] = "Yes"
if 'show_class_label' not in st.session_state:
    st.session_state['show_class_label'] = 'Show Labels'
if 'box_type' not in st.session_state:
    st.session_state['box_type'] = "regular"
if 'private_api_key' not in st.session_state:
    st.session_state['private_api_key'] = "FoM7U0Xonm0lSnV74Bfl"
if 'uploaded_file_od' not in st.session_state:
    st.session_state['uploaded_file_od'] = ""
if 'pill_input_array' not in st.session_state:
    st.session_state['pill_input_array'] = []
if 'box_width' not in st.session_state:
    st.session_state['box_width'] = "1"

color_choices = [
    (199,252,0),
    (255,0,255),
    (134,34,255),
    (254,0,86),
    (0,255,206),
    (255,128,0)
]
##########
#### Set up main app logic
##########

def run_inference(workspace_id, model_id, version_number, uploaded_img, inferenced_img):
    rf = Roboflow(api_key=st.session_state['private_api_key'])
    project = rf.workspace(workspace_id).project(model_id)
    project_metadata = project.get_version_information()
    version = project.version(version_number)
    model = version.model
    
    #TODO make sure I can remove this if statement
    if project.type != "object-detection":
        st.write("### Please include the project URL for an object detection model trained with Roboflow Train")

   
                
    predictions = model.predict(uploaded_img, overlap=int(st.session_state['overlap_threshold']),
        confidence=int(confidence_threshold), stroke=int(st.session_state['box_width']))

    predictions_json = predictions.json()

    # drawing bounding boxes with the Pillow library
    # https://docs.roboflow.com/inference/hosted-api#response-object-format
    collected_predictions = []
    collected_classes = []
    
    bounding_box_colors = {}

    for bounding_box in predictions:
        x0 = bounding_box['x'] - bounding_box['width'] / 2
        x1 = bounding_box['x'] + bounding_box['width'] / 2
        y0 = bounding_box['y'] - bounding_box['height'] / 2
        y1 = bounding_box['y'] + bounding_box['height'] / 2
        class_name = bounding_box['class']
        confidence_score = bounding_box['confidence']
        box = (x0, x1, y0, y1)
        detected_x = int(bounding_box['x'] - bounding_box['width'] / 2)
        detected_y = int(bounding_box['y'] - bounding_box['height'] / 2)
        detected_width = int(bounding_box['width'])
        detected_height = int(bounding_box['height'])
        # ROI (Region of Interest), or detected bounding box area
        roi_bbox = [detected_y, detected_height, detected_x, detected_width]
        collected_classes.append(class_name)
        collected_predictions.append({"class":class_name, "confidence":confidence_score,
                                    "x0,x1,y0,y1":[int(x0),int(x1),int(y0),int(y1)],
                                    "Width":int(bounding_box['width']), "Height":int(bounding_box['height']),
                                    "ROI, bbox (y+h,x+w)":roi_bbox,
                                    "bbox area (px)":abs(int(x0)-int(x1))*abs(int(y0)-int(y1))})
        # position coordinates: start = (x0, y0), end = (x1, y1)
        # color = RGB-value for bounding box color, (0,0,0) is "black"
        # thickness = stroke width/thickness of bounding box
        start_point = (int(x0), int(y0))
        end_point = (int(x1), int(y1))
        
        # draw/place bounding boxes on image
        cv2.rectangle(inferenced_img, start_point, end_point, color=(0,0,0), thickness=2)

        # add class name with filled background
        #cv2.rectangle(inferenced_img, (int(x0), int(y0)), (int(x0) + 40, int(y0) - 20), color=(0,0,0),
                        #thickness=-1)

        # draw/place bounding boxes on image
        #cv2.rectangle(inferenced_img, start_point, end_point, color=(0,0,0), thickness=1)

        # add class name with filled background
        #cv2.rectangle(inferenced_img, (int(x0), int(y0)), (int(x0) + 40, int(y0) - 20), color=(0,0,0),
                       # thickness=-1)
        

        cv2.putText(inferenced_img,
            class_name,#text to place on image
            (int(x0), int(y0) - 5),#location of text
            fontFace=cv2.FONT_HERSHEY_DUPLEX,#font
            fontScale=1,#font scale
            color=(255,255,255),#text color
            thickness=1#thickness/"weight" of text
            )
    
        
                

    ## Subtitle.
    st.write("### Inferenced Image")    
    st.image(inferenced_img, caption="Inferenced Image", use_column_width=True)

    results_tab = st.tabs(["Inference Results"])

#TODO only keep Results tab until done with setup and testing
    #with results_tab:
    ## Display results dataframe in main app.
    st.write('### Prediction Results (Pandas DataFrame)')
    st.dataframe(collected_predictions)
    
    #TODO this count will eventually move or be hidden. Need to add conditions to not count/compare if
    #confidence interval is too low. May get around this by having higher confidence interval anyways
    classesCount = Counter(collected_classes)
    st.write("Pill Images Recognized")
    st.write(classesCount)
    for pillsAnalyzed in collected_classes:
        st.write(pillsAnalyzed)
    st.write(" ")
    
    userInputCount = Counter(st.session_state['pill_input_array'])
    st.write("Pill Images Selected")
    st.write(userInputCount)
    for pillsSelected in st.session_state['pill_input_array']:
        st.write(pillsSelected)
    st.write(" ")

    st.write(len(classesCount))
    st.write(len(userInputCount))
    if len(collected_classes) != len(st.session_state['pill_input_array']): #len(classesCount) != len(userInputCount):
        print("Number of pills selected does not match the number of pills in the image. Please verify selections")

    
    else:
        flag=0
        for i in classesCount:
            if classesCount.get(i) != userInputCount.get(i):
                st.write("not a match")
                st.write(classesCount.get(i))
                st.write(userInputCount.get(i))
                st.write(" ")
                print(i)
                #st.write("For " + userInputCount. + ", you selected")
                flag=1
                break
        if flag==0:
            print("equal")
            st.write(" ")
            st.write("The correct pills have been chosen")
            
            #st.write("For "+"d You selected ")
        else:
            print("not equal")
            st.write(" ")
            st.write("The incorrect pills are chosen. Please try again")
            for x in userInputCount:
                print("expected " + x + " to have " + str(userInputCount.get(x)) + " and detected " + str(classesCount.get(x)))
                st.write("expected " + x + " to have " + str(userInputCount.get(x)) + " and detected " + str(classesCount.get(x)))


        

    

##########
##### Set up sidebar.
##########
# Add in location to select image.
with st.sidebar:
    st.write("#### Select an image to upload.")
    uploaded_file_od = st.file_uploader("Image File Upload",
                                        type=["png", "jpg", "jpeg"],
                                        accept_multiple_files=False)


    ## Add in sliders.
    confidence_threshold = st.slider("Confidence threshold (%): What is the minimum acceptable confidence level for displaying a bounding box?", 0, 100, 40, 1)
    overlap_threshold = st.slider("Overlap threshold (%): What is the maximum amount of overlap permitted between visible bounding boxes?", 0, 100, 30, 1)
    
    
    
    #holds the pill and qty of the pill
    pill_selected = ""
    pill_qty = ""

    pill_selected = st.selectbox("Pill Selector:",
                                 options=("8 HR Acetaminophen 650 MG Extended Release Tablet",
                                          "Acetaminophen 325 MG Tablet",
                                          "Acetaminophen 500 MG Tablet",
                                          "Albuterol 2 MG Tablet",
                                          "Albuterol 4 MG Tablet",
                                          "Amlodipine 2.5 MG Tablet",
                                          "Amlodipine 5 MG Tablet",
                                          "Amlodipine 10 MG Tablet",
                                          "Ibuprofen 200 MG Tablet",
                                          "Ibuprofen 400 MG Tablet",
                                          "Ibuprofen 600 MG Tablet",
                                          "Ibuprofen 800 MG Tablet",
                                          "Lisinopril 2.5 MG Tablet",
                                          "Lisinopril 5 MG Tablet",
                                          "Lisinopril 10 MG Tablet",
                                          "Lisinopril 20 MG Tablet",
                                          "Lisinopril 30 MG Tablet",
                                          "Lisinopril 40 MG Tablet",),
                                key="pill_selected")
    
    pill_qty = st.number_input("Enter the pill quantity", step=1,min_value=1,value=1)

    if st.button("Add a Pill and Qty"):
        x = pill_qty
        while x > 0:
            st.session_state['pill_input_array'].append(pill_selected)
            x=x-1
        print(st.session_state['pill_input_array'])
        pill_selected = ""
        pill_qty = ""

    if st.button("Clear Pill Entries"):
        st.session_state["pill_input_array"].clear()
    
    #st.write(st.session_state['pill_input_array'])

        
##########
##### Set up project access.
##########

## Title.
st.write("# Pill Detection")

#TODO Cleanup the text in this section and hide the need for the Project URL and the Private API key
with st.form("project_access"):
    project_url_od = "https://app.roboflow.com/ms-cs-thesis/medication-identification-v2/2" #st.text_input("Project URL", key="project_url_od",
                                #help="Copy/Paste Your Project URL: https://docs.roboflow.com/python#finding-your-project-information-manually",
                                #placeholder="https://app.roboflow.com/workspace-id/model-id/version")
    private_api_key = "FoM7U0Xonm0lSnV74Bfl" #st.text_input("Private API Key", key="private_api_key", type="password",placeholder="Input Private API Key")
    submitted = st.form_submit_button("Verify Selected Medication")
    st.write("*** From the left side bar, upload an image (jpg, jpeg, or png) and select each pill you are scheduled to take and the quantity by using the dropdowns and the Add a Pill and Qty button. Your selections can be cleared by using the Clear Pill Entries button. ***")
    
#https://app.roboflow.com/ms-cs-thesis/medication-identification-v2/2
    if submitted:
        st.write("Loading results...")
        extracted_url = project_url_od.split("roboflow.com/")[1]
        #after split1 ['ms-cs-thesis/medication-identification-v2/2']
        if "model" in project_url_od.split("roboflow.com/")[1]:
            workspace_id = extracted_url.split("/")[0]
            model_id = extracted_url.split("/")[1]
            version_number = extracted_url.split("/")[3]
        elif "deploy" in project_url_od.split("roboflow.com/")[1]:
            workspace_id = extracted_url.split("/")[0]
            model_id = extracted_url.split("/")[1]
            version_number = extracted_url.split("/")[3]
        else:
            workspace_id = "ms-cs-thesis" #extracted_url.split("/")[0] # ms-cs-thesis
            model_id = "medication-identification-v2" #extracted_url.split("/")[1] # medication-identification-v2
            version_number = "2" #extracted_url.split("/")[2] # 2

if uploaded_file_od != None:
    # User-selected image.
    image = Image.open(uploaded_file_od)
    uploaded_img = np.array(image)
    inferenced_img = uploaded_img.copy()

    run_inference(workspace_id, model_id, version_number, uploaded_img, inferenced_img)
