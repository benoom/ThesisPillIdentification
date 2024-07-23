import streamlit as st
import base64
from PIL import Image
import cv2
import numpy as np
from roboflow import Roboflow
from collections import Counter

#original framework of this app provided by Roboflow: https://github.com/roboflow/streamlit-web-app/tree/main

#initialize variables
project_url_od, private_api_key, uploaded_file_od, pill_input_array = ("", "", "", [])

#initialize variables to manage the changing session states for Streamlit
if 'project_url_od' not in st.session_state:
    st.session_state['project_url_od'] = "https://app.roboflow.com/ms-cs-thesis/medication-identification-v3/3"
if 'workspace_id' not in st.session_state:
    st.session_state['workspace_id'] = 'ms-cs-thesis'
if 'model_id' not in st.session_state:
    st.session_state['model_id'] = 'medication-identification-v3'
if 'version_number' not in st.session_state:
    st.session_state['version_number'] = '3'
if 'confidence_threshold' not in st.session_state:
    st.session_state['confidence_threshold'] = "90"
if 'overlap_threshold' not in st.session_state:
    st.session_state['overlap_threshold'] = "50"
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

##########
#### Set up main app logic
##########

#custom function to handle the roboflow inference and results
def run_inference(uploaded_img, inferenced_img):
    rf = Roboflow(api_key=st.session_state['private_api_key'])
    project = rf.workspace(st.session_state['workspace_id']).project(st.session_state['model_id'])
    version = project.version(st.session_state['version_number'])
    model = version.model
    
    #utilize the roboflow function for running inference on an image            
    predictions = model.predict(uploaded_img, overlap=int(st.session_state['overlap_threshold']),
        confidence=int(st.session_state['confidence_threshold']), stroke=int(st.session_state['box_width']))

    #Initialize arrays for inferred prediction response and the respective model classes
    collected_predictions = []
    collected_classes = []
    
    # drawing bounding boxes with the Pillow library
    # https://docs.roboflow.com/inference/hosted-api#response-object-format
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
        

        cv2.putText(inferenced_img,
            class_name,#text to place on image
            (int(x0), int(y0) - 5),#location of text
            fontFace=cv2.FONT_HERSHEY_DUPLEX,#font
            fontScale=1.5,#font scale
            color=(255,255,255),#text color
            thickness=2#thickness/"weight" of text
            )
    

    ## Subtitle for inferred image.
    st.write("### Pills and Predicted Identification")    
    st.image(inferenced_img, caption="Pills and Predicted Identification", use_column_width=True)

    #Create counters for the number of classes detected and the number of classes entered by the user    
    classesCount = Counter(collected_classes)
    userInputCount = Counter(st.session_state['pill_input_array'])
    
    if len(collected_classes) != len(st.session_state['pill_input_array']):
        #purely a check of whether the number of pills for the two arrays match 
        st.write("Number of pills selected does not match the number of pills in the image. Please verify selections")        
        for x in classesCount:
            #a further check to see if the pills selected by the user match up with the pills inferred from the uploaded image
            #write to the app the results to indicate if the correct pills were chosen or if the user needs to get more or put a pill(s) back
            if userInputCount.get(x) == None:
                st.write("Expected 0 " + x + " and detected " + str(classesCount.get(x)) + ". Please put the pill(s) back.")
            else:
                if userInputCount.get(x) != classesCount.get(x):
                    st.write("Expected " + str(userInputCount.get(x)) + " " + x + " and detected " + str(classesCount.get(x)) + ". Please put " + str(abs(userInputCount.get(x)-classesCount.get(x))) + " pill(s) back")
                else:
                    st.write("Expected " + str(userInputCount.get(x)) + " " + x + " and detected " + str(classesCount.get(x)))
        st.write(" ")

    # if the number of pills selected by the user and the number of pills inferred match then a breakdown is written to the screen indicating
    # what was expected for the number of each selected pill and the number of each pill that was detected
    else:
        flag=0
        for i in classesCount:
            if classesCount.get(i) != userInputCount.get(i):
                st.write("Not a Match")    
                flag=1
                break
        if flag==0:
            st.write(" ")
            st.write("The correct pills have been chosen")
            for x in userInputCount:
                st.write("Expected " + str(userInputCount.get(x)) + " " + x + " and detected " + str(classesCount.get(x)))
        else:
            st.write(" ")
            st.write("The incorrect pills are chosen. Please try again")
            for x in userInputCount:
                st.write("Expected " + str(userInputCount.get(x)) + " " + x + " and detected " + str(classesCount.get(x)))

    #write out a list on the screen of the pills selected and the pills detected
    st.write(" ")
    st.write("Pill Images Recognized:")
    for pillsAnalyzed in collected_classes:
        st.write(pillsAnalyzed)
    st.write(" ")

    st.write("Pill Images Selected:")    
    for pillsSelected in st.session_state['pill_input_array']:
        st.write(pillsSelected)
    st.write(" ")    

##########
##### Set up sidebar.
##########
# Add in location to select image.
with st.sidebar:
    st.write("#### Select an image to upload.")
    uploaded_file_od = st.file_uploader("Image File Upload",
                                        type=["png", "jpg", "jpeg"],
                                        accept_multiple_files=False)

    
    #holds the pill and qty of the pill
    pill_selected = ""
    pill_qty = ""

    #sets up the selector box for the user to indicate which pills they are prescribed
    pill_selected = st.selectbox("Pill Selector:",
                                 options=("8 HR Acetaminophen 650 MG Extended Release Tablet",                                          
                                          "Acetaminophen 500 MG Tablet",                                          
                                          "Ibuprofen 200 MG Tablet",
                                          ),
                                key="pill_selected")
    
    #store the qty of each pill the user selects
    pill_qty = st.number_input("Enter the pill quantity", step=1,min_value=1,value=1)

    #handle the set up of a button to input the pill and qty of the pill
    if st.button("Add a Pill and Qty"):
        x = pill_qty
        while x > 0:
            st.session_state['pill_input_array'].append(pill_selected)
            x=x-1
        pill_selected = ""
        pill_qty = ""

    # button to clear all pill entries recorded to the app
    if st.button("Clear Pill Entries"):
        st.session_state["pill_input_array"].clear()
        
##########
##### Set up project access.
##########

## Title.
st.write("# Pill Detection")

with st.form("project_access"):
    #hold the project url hosted on Roboflow and the API key
    project_url_od = "https://app.roboflow.com/ms-cs-thesis/medication-identification-v3/3" 
    private_api_key = "FoM7U0Xonm0lSnV74Bfl" 
    #button to kick off the code for checking if an image has been uploaded and setting the project URL if needed
    submitted = st.form_submit_button("Verify Selected Medication")
    st.write("*** From the left side bar, upload an image (jpg, jpeg, or png) and select each pill you are scheduled to take and the quantity by using the dropdowns and the Add a Pill and Qty button. Your selections can be cleared by using the Clear Pill Entries button. ***")
    
#https://app.roboflow.com/ms-cs-thesis/medication-identification-v3/3
    if submitted:
        st.write("Loading results...")
        extracted_url = project_url_od.split("roboflow.com/")[1]
        #after split1 ['ms-cs-thesis/medication-identification-v3/3']
        
        if "model" in project_url_od.split("roboflow.com/")[1]:
            st.session_state['workspace_id'] = extracted_url.split("/")[0]
            st.session_state['model_id'] = extracted_url.split("/")[1]
            st.session_state['version_number'] = extracted_url.split("/")[3]
        elif "deploy" in project_url_od.split("roboflow.com/")[1]:
            st.session_state['workspace_id'] = extracted_url.split("/")[0]
            st.session_state['model_id'] = extracted_url.split("/")[1]
            st.session_state['version_number'] = extracted_url.split("/")[3]
        else:
            st.session_state['workspace_id'] = "ms-cs-thesis" #extracted_url.split("/")[0] # ms-cs-thesis
            st.session_state['model_id'] = "medication-identification-v3" #extracted_url.split("/")[1] # medication-identification-v3
            st.session_state['version_number'] = "3" #extracted_url.split("/")[2] # 3
        
        if uploaded_file_od != None:
            # User-selected image.
            image = Image.open(uploaded_file_od)
            #alter the image to align the image closer to what Roboflow performs on their website when testing images against a model
            uploaded_img = cv2.imdecode(np.frombuffer(uploaded_file_od.read(), np.uint8), 1)
            #convert image to an array
            inferenced_img = np.array(image)

            #call the custom inference function
            run_inference(uploaded_img, inferenced_img)

