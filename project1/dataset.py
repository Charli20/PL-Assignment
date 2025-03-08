import pandas as pd
import numpy as np
import os
from src.config import Config

config = Config()

class CreateData:
    
    @staticmethod
    def create_df():
        data = {
        'disease': ['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis', 'Drug Reaction', 'Peptic ulcer disease', 'AIDS', 'Diabetes ', 'Gastroenteritis', 'Bronchial Asthma', 'Hypertension ', 'Migraine', 'Cervical spondylosis',
        'Paralysis (brain hemorrhage)', 'Jaundice', 'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'hepatitis A','Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E', 'Alcoholic hepatitis', 'Tuberculosis', 'Common Cold',
        'Pneumonia', 'Dimorphic hemorrhoids (piles)', 'Heart attack', 'Varicose veins', 'Hypothyroidism', 'Hyperthyroidism','Hypoglycemia', 'Osteoarthritis', 'Arthritis', '(Vertigo) Paroxysmal Positional Vertigo', 'Acne', 'Urinary tract infection','Psoriasis', 'Impetigo'],
        
        'medications': ['Clotrimazole, Fluconazole', 'Cetirizine, Loratadine', 'Omeprazole, Ranitidine', 'Ursodeoxycholic acid','Antihistamines, Prednisone', 'Pantoprazole, Antacids', 'Tenofovir, Efavirenz', 'Metformin, Insulin',
        'Oral rehydration salts, Loperamide', 'Albuterol, Budesonide', 'Lisinopril, Amlodipine', 'Sumatriptan, Ibuprofen','Naproxen, Muscle relaxants', 'Alteplase, Anticonvulsants', 'Phenobarbital, Antibiotics (if needed)',
        'Chloroquine, Artemisinin', 'Acyclovir, Calamine lotion', 'Paracetamol, IV fluids', 'Ciprofloxacin, Azithromycin','Supportive care, Rest', 'Tenofovir, Entecavir', 'Sofosbuvir, Ledipasvir', 'Peginterferon, Ribavirin',
        'Supportive care, Rest', 'Corticosteroids, Pentoxifylline', 'Rifampicin, Isoniazid', 'Pseudoephedrine, Dextromethorphan','Amoxicillin, Levofloxacin', 'Hydrocortisone cream, Pain relievers', 'Aspirin, Metoprolol','Diosmin, Ibuprofen', 'Levothyroxine', 'Methimazole, Propranolol', 'Glucose tablets, Glucagon',
        'Acetaminophen, Celecoxib', 'Methotrexate, Ibuprofen', 'Betahistine, Meclizine', 'Benzoyl peroxide, Doxycycline','Nitrofurantoin, Trimethoprim', 'Clobetasol, Methotrexate', 'Mupirocin, Cephalexin'],
        
        'diets': ['High-fiber, avoid sugary foods', 'Avoid allergens (e.g., nuts, pollen), high-vitamin C','Low-acid, avoid spicy foods, small meals', 'Low-fat, high-fiber', 'Avoid trigger substances, balanced diet',
          'High-protein, avoid spicy/fatty foods', 'High-calorie, nutrient-dense, balanced', 'Low-carb, high-fiber, consistent meals','Hydration, bland diet (e.g., bananas, rice)', 'Avoid triggers (e.g., smoke), anti-inflammatory foods',
          'Low-sodium, high-potassium (e.g., bananas)', 'Avoid caffeine/alcohol, hydration','Anti-inflammatory (e.g., fish, nuts)', 'Soft foods, high-calorie, vitamin D', 'Low-fat, high-protein, hydration',
          'High-protein, hydration, avoid fasting', 'Light diet, hydration, avoid heavy meals','Hydration, nutrient-rich fluids (e.g., broth)', 'High-calorie, bland, avoid raw foods','High-calorie, low-fat, avoid alcohol', 'High-protein, low-fat, avoid alcohol', 'High-protein, low-fat, avoid alcohol',
          'High-protein, low-fat, avoid alcohol', 'High-protein, low-fat, hydration', 'Low-fat, vitamin B-rich, no alcohol','High-protein, high-calorie, balanced', 'Warm fluids, vitamin C (e.g., citrus)', 'Hydration, easy-to-digest foods',
          'High-fiber, plenty of water', 'Low-fat, low-sodium (e.g., Mediterranean)', 'High-fiber, avoid tight clothing','High-fiber, iodine-rich (e.g., seafood)', 'Low-iodine, high-calcium', 'Frequent small meals, balanced carbs','Anti-inflammatory, vitamin D (e.g., dairy)', 'Anti-inflammatory, omega-3 (e.g., salmon)', 'Hydration, avoid caffeine',
          'Low-glycemic, avoid dairy', 'High-fluid, cranberry juice', 'Anti-inflammatory, avoid processed foods','Hydration, avoid greasy foods'],

        'workout': ['Light stretching, low-impact exercises (avoid excessive sweating)', 'Breathing exercises, yoga (helps with nasal passage relife)','Gentle walking, yoga, avoid intense physical activity post-meal', 'Low-impact exercises like walking or swimming','Mild stretches and gentle movements (depending on symptoms)', 'Gentle walking, avoid strenuous activities','Moderate intensity exercises such as walking, cycling, swimming', 'Aerobic exercises (e.g., walking, swimming, cycling)','Rest and hydration, light stretching when feeling better', 'Breathing exercises, low-intensity walking','Cardiovascular exercises like brisk walking, cycling, swimming', 'Relaxation techniques, light yoga, meditation','Neck stretches, gentle range-of-motion exercises', 'Physical therapy, strength training exercises', 'Low-intensity walking, yoga for liver health','Rest and hydration, light stretching once symptoms subside', 'Gentle yoga, walking (when fever reduces)', 'Rest, gentle stretching, avoid strenuous activity',
          'Light walking, stretching once recovery progresses', 'Rest, low-intensity exercises when liver function improves', 'Gentle walking, avoid high-intensity exercises', 'Low-impact exercises such as walking or swimming','Rest, avoid intense exercises, light stretching','Rest, gentle walking once symptoms subside', 'Rest, avoid high-intensity workouts, gentle exercises', 'Walking, stretching (as strength returns)',
          'Light walking, gentle stretching (if symptoms permit)', 'Rest, breathing exercises, light walking after recovery','Gentle walking, pelvic floor exercises', 'Light walking, gradual intensity increase after recovery','Low-impact exercises like walking or swimming', 'Aerobic exercises like walking, cycling, swimming', 'Moderate-intensity exercises (walking, swimming)',
          'Light exercises, regular walking, and balance activities', 'Joint mobility exercises, swimming, walking', 'Low-impact exercises like swimming, cycling, and stretching','Balance exercises, Tai Chi, yoga', 'Low-impact activities such as yoga, walking, swimming', 'Rest, light walking after symptoms improve','Moderate exercises like swimming, stretching, yoga', 'Rest, light stretching, avoid excessive sweating']}
        

        df4 = pd.DataFrame(data)

        return df4

    def merge_df(config, df4):
        df1 = pd.read_csv(config.DS1_PATH)
        df2 = pd.read_csv(config.DS2_PATH)

        df1.columns = df1.columns.str.lower()
        df2.columns = df2.columns.str.lower()

        new_df = pd.merge(df1, df2, on='disease', how='inner')
        new_df = pd.merge(new_df, df4, on='disease', how='inner')

      
        save_dir = os.path.dirname(config.NEW_DATA_PATH)
        os.makedirs(save_dir, exist_ok=True)

        new_df.to_csv(config.NEW_DATA_PATH, index=False)

        return new_df


print("\nCreating a new dataset")
df4 = CreateData.create_df()
new_df = CreateData.merge_df(config, df4)
print("\nThe new dataset is are finished creating")
   