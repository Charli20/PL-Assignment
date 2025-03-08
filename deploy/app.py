import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import webbrowser

image_paths = ["assets/image1.png", "assets/image2.png", "assets/image3.jpg", "assets/image5.jpg"]

df = pd.read_csv(r'C:\Users\chali\OneDrive\Documents\planing computer project\project1\datasets\created\update_df.csv')
df1 = pd.read_csv(r'C:\Users\chali\OneDrive\Documents\planing computer project\project1\datasets\created\new_df.csv')

bsm = tf.keras.models.load_model(r'C:\Users\chali\OneDrive\Documents\planing computer project\project1\model\best_model.keras')
bsm.summary()

bmd = r"C:\Users\chali\OneDrive\Documents\planing computer project\project2\model\bestmodel.keras"
ml = tf.keras.models.load_model(bmd)

cl= ["glioma", "meningioma", "notumor", "pituitary"]


symptoms_list = {symptom: i for i, symptom in enumerate(['fast_heart_rate',' mucoid_sputum',' pus_filled_pimples',' pain_in_anal_region',' back_pain',' altered_sensorium',' neck_pain',' cold_hands_and_feets',' bloody_stool',' breathlessness',
 ' extra_marital_contacts',' visual_disturbances',' stomach_bleeding',' yellowing_of_eyes',' stiff_neck',' small_dents_in_nails',' bladder_discomfort',' increased_appetite',
 ' runny_nose',' mild_fever',' weight_gain',' swelling_of_stomach',' obesity',' shivering',' anxiety',' pain_behind_the_eyes',' blood_in_sputum',' drying_and_tingling_lips',
 ' palpitations',' family_history',' dehydration',' silver_like_dusting',' acidity',' muscle_wasting',' bruising',' mood_swings',' prominent_veins_on_calf',' blurred_and_distorted_vision',' inflammatory_nails',' yellow_crust_ooze',' fluid_overload',
 ' loss_of_balance',' spotting_ urination',' fatigue',' irritability',' receiving_blood_transfusion',' diarrhoea',' excessive_hunger',' swollen_blood_vessels',' red_sore_around_nose',' malaise',
 ' cough',' unsteadiness',' muscle_pain',' puffy_face_and_eyes',' depression',' scurring',' painful_walking',' red_spots_over_body',' burning_micturition',' lack_of_concentration',
 ' receiving_unsterile_injections',' throat_irritation',' muscle_weakness',' swelling_joints',' passage_of_gases',' continuous_feel_of_urine',' chills',' irregular_sugar_level',' blister',' skin_peeling',' stomach_pain',' sweating',' coma',
 ' brittle_nails',' yellow_urine',' history_of_alcohol_consumption',' lethargy',' abdominal_pain',' enlarged_thyroid',' blackheads',' belly_pain',' watering_from_eyes',' abnormal_menstruation',' loss_of_appetite',' sunken_eyes',' dark_urine',' irritation_in_anus',' weakness_in_limbs',' phlegm',' vomiting',' constipation',' foul_smell_of urine',' ulcers_on_tongue',' spinning_movements',' sinus_pressure',' nausea',' hip_joint_pain',' dizziness',' continuous_sneezing',' nodal_skin_eruptions',
 ' chest_pain',' high_fever',' internal_itching',' indigestion',' weakness_of_one_body_side',' toxic_look_(typhos)',' redness_of_eyes',' rusty_sputum',' joint_pain','itching',' knee_pain',' patches_in_throat',' dischromic _patches',' loss_of_smell',' swollen_extremeties',' weight_loss',' congestion',' polyuria',' yellowish_skin',' pain_during_bowel_movements',
 ' cramps',' skin_rash',' movement_stiffness',' slurred_speech',' swelled_lymph_nodes',' swollen_legs',' distention_of_abdomen',' acute_liver_failure',' restlessness',' headache'])}


le = LabelEncoder()
le.fit(df['Disease'])
df['Disease'] = df['Disease'].str.lower()
df.columns = df.columns.str.lower()
df1.columns = df1.columns.str.lower()

def predict_disease(selected_symptoms):
    input_vector = np.zeros(len(symptoms_list))
    
 
    for symptom in selected_symptoms:
        if symptom in symptoms_list:
            input_vector[symptoms_list[symptom]] = 1   
    input_vector = input_vector.reshape(1, 131, 1)  #
    predicted_disease_label = bsm.predict(input_vector).argmax(axis=1)[0]
    predicted_disease = le.inverse_transform([predicted_disease_label])[0]
    return predicted_disease


def disease_details(predicted_ds):
    des_info = df1[df1['disease'] == predicted_ds]['description']
    des_info = " ".join([w for w in des_info.values])

    pre = df1[df1['disease'] == predicted_ds][['precaution_1','precaution_2','precaution_3','precaution_4']]
    if pre.empty:
        pre = [["No precautions available."] * 4] 
    else:
        pre = pre.values
    med = df1[df1['disease'] == predicted_ds]['medications']
    med = " ".join([med for med in med.values])

    diet = df1[df1['disease'] == predicted_ds]['diets']
    diet = " ".join([diet for diet in diet.values])

    wor = df1[df1['disease'] == predicted_ds]['workout']
    wor = " ".join([wor for wor in wor.values])

    return des_info, pre, med, diet, wor


def preprocess_image(img):
    img = img.resize((150, 150)) 
    img = np.array(img) 
    img = img.astype('float32') / 255  
    img = np.expand_dims(img, axis=0)
    return img

def predict_tumor(img):
    processed_img = preprocess_image(img)
    predictions = ml.predict(processed_img)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return categories[predicted_class]

def display_introduction():
    st.markdown("""
        <style>
            body {
                background: linear-gradient(to right, #6a11cb 0%, #2575fc 100%);
                color: white;
                font-family: Arial, sans-serif;
            }
            .big-title {
                font-size: 40px;
                font-weight: bold;
                color: #2c3e50;
                text-align: center;
            }
            .big-intro {
                font-size: 20px;
                font-weight: 400;
                color: #34495e;
                text-align: center;
            }
            .stButton > button {
                font-size: 20px;
                padding: 20px;
                width: 100%;
            }
            .gradient-bg {
                padding: 20px;
                border-radius: 15px;
            }
            .container {
                padding: 10px;
            }
        </style>
    """, unsafe_allow_html=True)
    
    
    st.markdown('<h1 class="big-title">Welcome to HealthTech Innovations</h1>', unsafe_allow_html=True)
    st.markdown('<p class="big-intro">HealthTech Innovations is a leading healthcare technology company focused on transforming '
                'patient care with Big Data and AI solutions. Our mission is to leverage cutting edge technologies '
                'to improve diagnostic accuracy, enhance personalized treatment plans, and drive innovation in healthcare.</p>', unsafe_allow_html=True)
    
    st.write("---")


def display_images():
    col1, col2, col3 = st.columns(3)
    
    with col1:
        img = Image.open(image_paths[0]) 
        img = img.resize((500, 500)) 
        st.image(img, use_container_width=True)
        
    with col2:
        img = Image.open(image_paths[1])
        img = img.resize((500, 500)) 
        st.image(img, use_container_width=True)
        
    with col3:
        img = Image.open(image_paths[2]) 
        img = img.resize((500, 500)) 
        st.image(img, use_container_width=True)


def display_footer():
    st.write("---")
    st.write(f"### About Me")
    st.write("""Hello, I'm Chalidu Bandara Wijekoon. This is my planning computer project assignment.""")
        
    st.write("Connect with me on social media:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('[LinkedIn](https://www.linkedin.com/in/chalidu-wijekoon-656118262)', unsafe_allow_html=True)
        
    with col2:
        st.markdown('[GitHub](https://github.com/Charli20)', unsafe_allow_html=True)

    with col3:
        st.markdown('[Instagram](https://www.instagram.com/_chaliya___20/?igsh=MW1jNDN3cTR3M3I3Yg%3D%3D)', unsafe_allow_html=True)

def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Choose an option:", ["Home", "Brain Tumor Detection", "Disease Prediction"])

    if 'page' not in st.session_state:
        st.session_state.page = "home"
        
    if selection == "Home":
        st.session_state.page = "home"
        display_introduction()
        display_images()
        display_footer()

    elif selection == "Brain Tumor Detection":
        st.session_state.page = "brain_tumor_detection"
        st.markdown('<h1 class="main-title">Brain Tumor Detection</h1>', unsafe_allow_html=True)
        st.write("Upload a brain MRI image to detect whether it contains a brain tumor or not.")

        uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded MRI Image", use_container_width=True)
            img = Image.open(uploaded_file)
            img = img.resize((150, 150))  
            img_array = img_to_array(img) 
            img_array = np.expand_dims(img_array, axis=0)  
            img_array = img_array / 255.0 

            predictions = ml.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)[0]  

            if cl[predicted_class] == "notumor":
                st.write("No tumor detected.")
            else:
                st.write(f"Prediction: {cl[predicted_class]}")
                st.write(f"Confidence: {predictions[0][predicted_class]*100:.2f}%")

    elif selection == "Disease Prediction":
        st.session_state.page = "disease_prediction"
        st.markdown('<h1 class="main-title">Disease & Symptoms Prediction</h1>', unsafe_allow_html=True)
        st.write("Select symptoms to predict the disease:")

        selected_symptoms = st.multiselect("Select Symptoms", list(symptoms_list.keys()))

        if st.button("Predict Disease", key="predict_button"):
            if selected_symptoms:
                predicted_disease = predict_disease(selected_symptoms)
                st.subheader(f"Predicted Disease: {predicted_disease}")
                des_info, pre, med, diet, wor = disease_details(predicted_disease)

                st.write("**Disease Description:**")
                st.write(des_info)

                st.write("**Precautions:**")
                for i, precaution in enumerate(pre[0]):
                    st.write(f"{i+1}. {precaution}")

                st.write("Medication:")
                st.write(med)

                st.write("Diets:")
                st.write(diet)

                st.write("**Workout Recommendations:**")
                st.write(wor)
            else:
                st.error("Please select at least one symptom!")

if __name__ == "__main__":
    main()