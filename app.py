
import streamlit as st
import pickle
from streamlit_option_menu import option_menu
from PIL import Image
import keras
from PIL import Image, ImageOps
import numpy as np


def func(img, weights_file):
    # Load the model
    model = keras.models.load_model(weights_file)

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224,3), dtype=np.float32)
    image = img
    #image sizing
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    return np.argmax(prediction)



st.set_page_config(page_title="Disease Detection and Prediction System")
diabetes_model=pickle.load(open('diabetes_model.sav','rb'))
heart_disease_model=pickle.load(open('heart_disease_model.sav','rb'))

with st.sidebar:
    selected=option_menu('Disease Detection & Prediction System',
    ['Disease Detection',
    'Disease Prediction'],
    icons=['search','bar-chart-line-fill'],
    default_index=0)


#******************************************DISEASE DETECTION******************************************************


if(selected=='Disease Detection'):
    bbb=st.sidebar.selectbox("Select a Disease",("Breast Cancer","Brain Tumor"))
    if(bbb=="Breast Cancer"):
        st.title("Breast Cancer Detection")
        st.text("Upload an Ultrasound scan for Detection")


        uploaded_file = st.file_uploader("Choose a scan ...", type=["png","jpg","jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Scan.', use_column_width=True)
            st.write("")
            
            label = func(image, 'breast_cancer.h5')
            
            if label == 0:
                st.success("The scan is normal")
            elif label == 1:
                st.success("The scan is benign")
            else:
                st.success("The scan is malignant")

    



    if(bbb=="Brain Tumor"):
        st.title("Brain Tumor Detection")
        st.text("Upload an MRI scan for Detection")


        uploaded_file = st.file_uploader("Choose a scan ...", type=["jpg","jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Scan.', use_column_width=True)
            st.write("")
            
            label = func(image, 'brain_tumor.h5')
            
            if label == 0:
                st.success("The scan is normal")
    
            else:
                st.success("The scan shows presence of brain tumor")

    



#************************************************DISEASE PREDICTION**************************************************


if(selected=='Disease Prediction'):
    
    aaa=st.sidebar.selectbox("Select a Disease",("Diabetes","Heart Disease"))
    if(aaa=="Diabetes"):
        st.title("Diabetes Prediction")
        Pregnancies = st.number_input('Number of pregnancies',step=1)
        Glucose = st.slider('Glucose Level',step=1,min_value=0,max_value=200)
        BloodPressure = st.slider('Blood Pressure Value',min_value=0,max_value=130,step=1)
        SkinThickness = st.slider('Skin Thickness Value ',min_value=0,max_value=100,step=1)
        Insulin = st.slider('Insulin Level',min_value=0,max_value=850,step=1)
        BMI = st.number_input('BMI Value',min_value=0.0,max_value=70.0,step=0.1)
        DiabetesPedigreeFunction = st.slider('Diabetes Pedigree Function Value',min_value=0.0,max_value=2.5)
        Age = st.number_input('Age of Person',min_value=0,max_value=100,step=1)

        diab_diagnosis=''

        if st.button("Result"):
            diab_prediction=diabetes_model.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])

            if(diab_prediction[0]==1):
                diab_diagnosis = 'The person is Diabetic'
        
            else:
                diab_diagnosis='The person is not Diabetic'
    
        st.success(diab_diagnosis)

#*###########################################################################################################
#############################################################################################################
    

    if(aaa=="Heart Disease"):
        st.title("Heart Disease Prediction")
        Age = st.number_input('Enter age',min_value=0,max_value=120,step=1)
        g=st.selectbox('Gender',('Male','Female'))
        if(g=="Male"):
            Gender=1
        else:
            Gender=0
        
        cpt = st.selectbox('Chest Pain Type',('Typical Angina','Atypical Angina','Non-Anginal Pain','Asymptotic'))
        if(cpt=='Typical Angina'):
            Chest_Pain_Type=0
        elif(cpt=='Atypical Angina'):
            Chest_Pain_Type=1
        elif(cpt=='Non-Anginal Pain'):
            Chest_Pain_Type=2
        else:
            Chest_Pain_Type=3
        

        Resting_bp = st.slider('Resting Blood Pressure in mm Hg ',min_value=80,max_value=200,step=1)
        
        Serum_Cholestrol = st.slider('Serum Cholestrol in mg/dl',min_value=120,max_value=570,step=1)
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl',('Yes','No'))
        if(fbs=='Yes'):
            Fasting_Blood_Sugar=1
        else:
            Fasting_Blood_Sugar=0


        rer = st.selectbox('Resting ECG Result',('Normal','Having ST-T wave abnormality','Left ventricular hypertrophy'))
        if(rer=='Normal'):
            Resting_ECG_Result=0
        elif(rer=='Having ST-T wave abnormality'):
            Resting_ECG_Result=1
        else:
            Resting_ECG_Result=2

        Maximum_heartrate = st.slider('Maximum Heart Rate Achieved',min_value=70,max_value=210,step=1)
        eia = st.selectbox('Exercise Induced Angina',('Yes','No'))
        if(eia=='Yes'):
            Exercise_Induced_Angina=1
        else:
            Exercise_Induced_Angina=0
        
        OldPeak = st.slider('Oldpeak: ST depression induced by exercise relative to rest',min_value=0.0,max_value=7.0,step=0.1)
        sl= st.selectbox('Slope of the peak exercise ST segment',('Upsloping','Flat','Downsloping'))
        if(sl=="Upsloping"):
            Slope=0
        elif(sl=="Flat"):
            Slope=1
        else:
            Slope=2
        
        Fluorosopy=st.slider('Number of major vessels colored by flourosopy',min_value=0,max_value=4,step=1)
        tl=st.selectbox('Thalassemia Value',('Normal','Fixed Defect','Reversible Defect'))
        if(tl=="Normal"):
            Thal=1
        elif(tl=="Fixed Defect"):
            Thal=2
        else:
            Thal=3


        hd_diagnosis=''

        if st.button("Result"):
            hd_prediction=heart_disease_model.predict([[Age,Gender,Chest_Pain_Type,Resting_bp,Serum_Cholestrol,Fasting_Blood_Sugar,Resting_ECG_Result,Maximum_heartrate,Exercise_Induced_Angina,OldPeak,Slope,Fluorosopy,Thal]])

            if(hd_prediction[0]==1):
                hd_diagnosis = 'The person has Heart Disease'
        
            else:
                hd_diagnosis='The person is Normal'
    
        st.success(hd_diagnosis)

#*###########################################################################################################
#############################################################################################################
    

    



