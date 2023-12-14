import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
import pytesseract
import re
from nltk.metrics import edit_distance
from nltk.tokenize import word_tokenize
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update with your actual path

df=pd.read_csv(r'df')
# Handling Image and converting to text

class ImageProcessor:
    def __init__(self):
        self.images = []

    def get_grayscale(self, image):
        image_np = np.array(image)

        return cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    def down(self, image):
        image_np = np.array(image)
        return cv2.resize(image_np, (0, 0), fx=1.5, fy=1.5)

    def thresholding(self, image):
        image_np = np.array(image)

        return cv2.threshold(image_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    def match_template(self, image, template):
        image_np = np.array(image)
        return cv2.matchTemplate(image_np, template, cv2.TM_CCOEFF_NORMED)

    def image_to_text(self, file_path):
        # print(file_path)
        if file_path:
            img = Image.open(file_path)

            # # Open the image using OpenCV
            # img = cv2.imread(file_path)

            # Apply image enhancement functions
            img = self.get_grayscale(img)
            img = self.thresholding(img)
            img = self.down(img)

            # Convert the image to PIL format
            img_pil = Image.fromarray(img)

            # Perform OCR on the enhanced image
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(img_pil, config=custom_config)

            # Append the image data and extracted text to the list
            self.images.append({'image': img_pil, 'text': text})

            dict=self.regex_processing(self.images[0]['text'])
            
            for key, value in dict.items():
                    # Check if even a part of the key is present in df_columns
                    matching_column = self.most_close_column(key.lower(),df.columns)
                    print('key:',key, 'Value:',value,'COl:',matching_column)

                    # If there's a match, add the key's value to the corresponding column(s)
                    if matching_column :
                        df.at[0, matching_column] = value  # Assuming you want to add values to the first row of the DataFrame



            
    def regex_processing(self,text):
            pattern = re.compile(r'\b([A-Z][a-zA-Z\s | a-zA-Z \(\w+\)\s ]+)\s+([0-9]+(?:\.[0-9]+)? | [0-9]+(?:\.[0-9]+)?e?-?[0-9]+?)')
            matches = pattern.findall(text)
            dict1={}
            # Print the result
            for match in matches:
                key = match[0].strip()
                if key not in dict1:
                    dict1[key] = match[1]  # Initialize an empty list for the key if it doesn't exist
        
            self.images[0]['text']=dict1 


            return   self.images[0]['text']
    
    def most_close_column(self,key, df_columns, threshold=3):
        closest_column = None
        min_distance = float('inf')

        for col in df_columns:
            col_lower = col.lower()
            key_lower = key.lower()

            words1 = set(word_tokenize(col_lower))
            words2 = set(word_tokenize(key_lower))

            common_words = words1.intersection(words2)
            distance = edit_distance(col_lower, key_lower)

            if distance < min_distance or bool(common_words) and distance == min_distance:
                closest_column = col
                min_distance = distance

        return closest_column


        
image_processor = ImageProcessor()


def make_prediction():
    print(df.columns)
    prediction_mapping={
    0: 'Allergies or Parasitic Infection',
    1: 'Anemia',
    2: 'Aplastic Anemia',
    3: 'Chronic Leukemia',
    4: 'Folate or Vitamin B12 Deficiency',
    5: 'Inflammation or Infection',
    6: 'Iron Deficiency Anemia',
    7: 'Megaloblastic Anemia',
    8: 'Normal',
    9: 'Thalassemia',
    10: 'Thrombocytosis',
    11: 'Unknown'
}
    # Load the saved XGBoost model from the pickle file
    with open(r'trained_pipeline.pkl', 'rb') as model_file:
        loaded_pipeline = pickle.load(model_file)
    prediction=loaded_pipeline.predict(df.iloc[[0]])[0]  
    print(prediction_mapping[prediction]) 
    return prediction_mapping[prediction]
    


# Set page title and configure layout
st.set_page_config(
    page_title="Blood Report Analyzer",
    page_icon=":microscope:",
    layout="wide",
)

# Create a gradient background using a custom CSS style
background_color = "#f2f5f7"
gradient_color = "#bdddfb"
st.markdown(
    f"""
    <style>
        .gradient-background {{
            background: linear-gradient(to right, {background_color}, {gradient_color});
            padding: 10px;
            border-radius: 5px;
        }}
    </style>
    """,
    unsafe_allow_html=True
)


# Custom CSS for centering and adjusting image size
st.markdown(
    """
    <style>
        .centered-image {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 30px;
            width: 30px;
            margin: auto;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# st.image("https://png.pngtree.com/element_our/20190602/ourmid/pngtree-cartoon-medical-heart-rate-graph-illustration-image_1406720.jpg", width=150)
# Title and description
st.title("Analyse your Blood Reports here")

# Placeholder for further analysis results
left_co, cent_co,last_co = st.columns(spec=[1,2,1])


with cent_co:
    st.subheader("Project Description:")
    st.write(
        """
        This application allows you to upload your blood report images and analyze them.
        Upload your blood report image using the button below and get insights into your health.
        """
    )

    # Upload button for blood report image (accept only image files)
    uploaded_file = st.file_uploader(
        "Upload Blood Report Image",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )

    def center_image(image_path):
        st.markdown(
            f'<div style="display: flex; justify-content: center; align-items: center;">'
            f'<img src="{image_path}" style="width: auto; max-width: 100%; height: auto;">'
            f'</div>',
            unsafe_allow_html=True
        )

    
    if uploaded_file is not None:
        st.success("Blood report image uploaded successfully!")
        
        # Display the uploaded image
        image = Image.open(uploaded_file)


        file_path = image.info.get('filename', None)

        if uploaded_file:
            print("File Path:", uploaded_file)
        else:
            print("File path not found in image metadata.")

        image_processor.image_to_text(uploaded_file)
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=["Unnamed: 0"])

        result=make_prediction()
        st.write("Prediction Result:")
        st.text(result)
        with cent_co:
            st.image(image, caption="Uploaded Blood Report Image", use_column_width=False, width=500, )
        center_image(image)
        # Add further analysis code here


# Additional information
st.markdown("<br>", unsafe_allow_html=True)
st.subheader("Additional Information:")
st.write(
    """
    - The accepted file formats for blood report documents include JPG, JPEG, and PNG.
    - Ensure that your document is clear and legible for accurate analysis.
    - Download the converted PDF to review the analyzed insights of your blood report.
    """
)

# Footer and contact information
st.markdown("<br>", unsafe_allow_html=True)
st.text("Developed by Your Name")
st.text("Contact: your.email@example.com")

# Additional information or links can be added here
