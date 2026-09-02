import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

# 1. Configuración de la página
st.set_page_config(page_title="Motor de Triaje Predictivo", layout="wide")

st.title("🩺 Sistema de Triaje y Diagnóstico Predictivo")
st.markdown("Selecciona los síntomas observados para calcular la probabilidad clínica de las patologías en tiempo real.")

# 2. Diccionario de Traducción de Síntomas (Inglés <-> Español)
MAPA_SINTOMAS = {
    'itching': 'Picazón / Prurito',
    'skin_rash': 'Erupción cutánea / Salpullido',
    'nodal_skin_eruptions': 'Erupciones nodulares en la piel',
    'continuous_sneezing': 'Estornudos continuos',
    'shivering': 'Escalofríos con temblor',
    'chills': 'Escalofríos',
    'joint_pain': 'Dolor articular (Artralgia)',
    'stomach_pain': 'Dolor de estómago',
    'acidity': 'Acidez / Ardor de estómago',
    'ulcers_on_tongue': 'Úlceras / Llagas en la lengua',
    'muscle_wasting': 'Atrofia muscular',
    'vomiting': 'Vómitos',
    'burning_micturition': 'Ardor al orinar (Disuria)',
    'spotting_ urination': 'Goteo / Manchado al orinar',
    'spotting_urination': 'Goteo / Manchado al orinar',
    'fatigue': 'Fatiga / Cansancio',
    'weight_gain': 'Aumento de peso',
    'anxiety': 'Ansiedad',
    'cold_hands_and_feets': 'Manos y pies fríos',
    'mood_swings': 'Cambios de humor',
    'weight_loss': 'Pérdida de peso involuntaria',
    'restlessness': 'Inquietud / Agitación',
    'lethargy': 'Letargo / Somnolencia profunda',
    'patches_in_throat': 'Placas / Manchas en la garganta',
    'irregular_sugar_level': 'Nivel irregular de azúcar en sangre',
    'cough': 'Tos',
    'high_fever': 'Fiebre alta',
    'sunken_eyes': 'Ojos hundidos',
    'breathlessness': 'Falta de aire (Disnea)',
    'sweating': 'Sudoración excesiva',
    'dehydration': 'Deshidratación',
    'indigestion': 'Indigestión / Dispepsia',
    'headache': 'Dolor de cabeza (Cefalea)',
    'yellowish_skin': 'Piel amarillenta (Ictericia)',
    'dark_urine': 'Orina oscura',
    'nausea': 'Náuseas',
    'loss_of_appetite': 'Pérdida del apetito (Anorexia)',
    'pain_behind_the_eyes': 'Dolor retroocular (Detrás de los ojos)',
    'back_pain': 'Dolor de espalda',
    'constipation': 'Estreñimiento',
    'abdominal_pain': 'Dolor abdominal',
    'diarrhoea': 'Diarrea',
    'mild_fever': 'Fiebre moderada / Febrícula',
    'yellow_urine': 'Orina amarillenta',
    'yellowing_of_eyes': 'Ojos amarillentos (Ictericia ocular)',
    'acute_liver_failure': 'Insuficiencia hepática aguda',
    'fluid_overload': 'Sobrecarga de líquidos / Edema general',
    'swelling_of_stomach': 'Distensión / Hinchazón abdominal',
    'swelled_lymph_nodes': 'Ganglios linfáticos inflamados',
    'malaise': 'Malestar general',
    'blurred_and_distorted_vision': 'Visión borrosa y distorsionada',
    'phlegm': 'Flema / Mucosidad',
    'throat_irritation': 'Irritación de garganta',
    'redness_of_eyes': 'Ojos rojos / Congestión conjuntival',
    'sinus_pressure': 'Presión sinusal',
    'runny_nose': 'Goteo nasal (Rinorrea)',
    'congestion': 'Congestión nasal',
    'chest_pain': 'Dolor en el pecho',
    'weakness_in_limbs': 'Debilidad en las extremidades',
    'fast_heart_rate': 'Taquicardia / Pulso acelerado',
    'pain_during_bowel_movements': 'Dolor al evacuar',
    'pain_in_anal_region': 'Dolor en la región anal',
    'bloody_stool': 'Heces con sangre',
    'irritation_in_anus': 'Irritación anal',
    'neck_pain': 'Dolor de cuello',
    'dizziness': 'Mareo',
    'cramps': 'Calambres',
    'bruising': 'Hematomas / Cardenales',
    'obesity': 'Obesidad',
    'swollen_legs': 'Piernas hinchadas',
    'swollen_blood_vessels': 'Vasos sanguíneos inflamados',
    'puffy_face_and_eyes': 'Cara y ojos hinchados',
    'enlarged_thyroid': 'Bocio / Tiroides agrandada',
    'brittle_nails': 'Uñas quebradizas',
    'swollen_extremeties': 'Extremidades hinchadas',
    'excessive_hunger': 'Hambre excesiva (Polifagia)',
    'extra_marital_contacts': 'Relaciones sexuales de riesgo / Contactos extrapareja',
    'drying_and_tingling_lips': 'Labios secos y con hormigueo',
    'slurred_speech': 'Dificultad para articular palabras (Disartria)',
    'knee_pain': 'Dolor de rodilla',
    'hip_joint_pain': 'Dolor en la articulación de la cadera',
    'muscle_weakness': 'Debilidad muscular',
    'stiff_neck': 'Rigidez de cuello',
    'swelling_joints': 'Hinchazón articular',
    'movement_stiffness': 'Rigidez al moverse',
    'spinning_movements': 'Sensación de giro / Vértigo',
    'loss_of_balance': 'Pérdida del equilibrio',
    'unsteadiness': 'Inestabilidad al marchar',
    'weakness_of_one_body_side': 'Debilidad en un lado del cuerpo (Hemiparesia)',
    'loss_of_smell': 'Pérdida del olfato (Anosmia)',
    'bladder_discomfort': 'Molestia en la vejiga',
    'foul_smell_of urine': 'Orina con mal olor',
    'foul_smell_of_urine': 'Orina con mal olor',
    'continuous_feel_of_urine': 'Sensación continua de orinar',
    'passage_of_gases': 'Flatulencias / Gases',
    'internal_itching': 'Picazón interna',
    'toxic_look_(typhos)': 'Aspecto tóxico / Tífico',
    'depression': 'Depresión',
    'irritability': 'Irritabilidad',
    'muscle_pain': 'Dolor muscular (Mialgia)',
    'altered_sensorium': 'Alteración del nivel de conciencia',
    'red_spots_over_body': 'Manchas rojas en el cuerpo',
    'belly_pain': 'Dolor de vientre',
    'abnormal_menstruation': 'Menstruación anormal / Irregular',
    'dischromic _patches': 'Parches / Manchas despigmentadas o discrómicas',
    'dischromic_patches': 'Parches / Manchas despigmentadas o discrómicas',
    'watering_from_eyes': 'Lagrimeo constante (Epífora)',
    'increased_appetite': 'Aumento del apetito',
    'polyuria': 'Micción frecuente (Poliuria)',
    'family_history': 'Antecedentes familiares de enfermedad',
    'mucoid_sputum': 'Esputo / Flema mucoide',
    'rusty_sputum': 'Esputo herrumbroso (Rojizo / Marrón)',
    'lack_of_concentration': 'Falta de concentración',
    'visual_disturbances': 'Alteraciones visuales',
    'receiving_blood_transfusion': 'Historial de transfusión sanguínea',
    'receiving_unsterile_injections': 'Uso de inyecciones no estériles',
    'coma': 'Coma / Estado comatoso',
    'stomach_bleeding': 'Sangrado estomacal / Digestivo',
    'distention_of_abdomen': 'Distensión abdominal',
    'history_of_alcohol_consumption': 'Historial de consumo de alcohol',
    'blood_in_sputum': 'Esputo con sangre (Hemoptisis)',
    'prominent_veins_on_calf': 'Venas prominentes en las pantorrillas (Varices)',
    'palpitations': 'Palpitaciones',
    'painful_walking': 'Dolor al caminar',
    'pus_filled_pimples': 'Granos con pus (Pústulas)',
    'blackheads': 'Espinillas / Puntos negros',
    'scurring': 'Cicatrices / Descamación cutánea',
    'skin_peeling': 'Descamación de la piel',
    'silver_like_dusting': 'Descamación plateada en piel',
    'small_dents_in_nails': 'Pequeñas hendiduras en las uñas (Pitting)',
    'inflammatory_nails': 'Inflamación en las uñas (Paroniquia)',
    'blister': 'Ampollas',
    'red_sore_around_nose': 'Llagas rojas alrededor de la nariz',
    'yellow_crust_ooze': 'Costras amarillentas con secreción'
}

# 3. Diccionario de Traducción de Enfermedades / Diagnósticos (Actualizado con valores exactos)
MAPA_ENFERMEDADES = {
    'Fungal infection': 'Infección por Hongos',
    'Allergy': 'Alergia',
    'GERD': 'Reflujo Gastroesofágico (ERGE)',
    'Chronic cholestasis': 'Colestasis Crónica',
    'Drug Reaction': 'Reacción Adversa a Medicamentos',
    'Peptic ulcer disease': 'Úlcera Péptica',
    'AIDS': 'SIDA',
    'Diabetes': 'Diabetes Mellitus',
    'Gastroenteritis': 'Gastroenteritis',
    'Bronchial Asthma': 'Asma Bronquial',
    'Hypertension': 'Hipertensión Arterial',
    'Migraine': 'Migraña',
    'Cervical spondylosis': 'Espondilosis Cervical',
    'Paralysis (brain hemorrhage)': 'Parálisis (Hemorragia Cerebral)',
    'Jaundice': 'Ictericia',
    'Malaria': 'Paludismo / Malaria',
    'Chicken pox': 'Varicela',
    'Dengue': 'Dengue',
    'Typhoid': 'Fiebre Tifoidea',
    'hepatitis A': 'Hepatitis A',
    'Hepatitis B': 'Hepatitis B',
    'Hepatitis C': 'Hepatitis C',
    'Hepatitis D': 'Hepatitis D',
    'Hepatitis E': 'Hepatitis E',
    'Alcoholic hepatitis': 'Hepatitis Alcohólica',
    'Tuberculosis': 'Tuberculosis',
    'Common Cold': 'Resfriado Común',
    'Pneumonia': 'Neumonía',
    'Dimorphic hemorrhoids(piles)': 'Hemorroides',
    'Heart attack': 'Infarto Agudo de Miocardio',
    'Varicose veins': 'Varices',
    'Hypothyroidism': 'Hipotiroidismo',
    'Hyperthyroidism': 'Hipertiroidismo',
    'Hypoglycemia': 'Hipoglucemia',
    'Osteoarthritis': 'Osteoartritis',
    'Arthritis': 'Artritis',
    'Benign Paroxysmal Positional Vertigo': 'Vértigo Posicional Paroxístico Benigno',
    'Acne': 'Acné',
    'Urinary tract infection': 'Infección del Tracto Urinario (ITU)',
    'Psoriasis': 'Psoriasis',
    'Impetigo': 'Impétigo'
}

# 4. Funciones limpiadoras para evitar fallas por espacios extra en las cadenas
def obtener_traduccion_sintoma(sintoma_raw):
    sintoma_clean = sintoma_raw.strip()
    if sintoma_clean in MAPA_SINTOMAS:
        return MAPA_SINTOMAS[sintoma_clean]
    if sintoma_raw in MAPA_SINTOMAS:
        return MAPA_SINTOMAS[sintoma_raw]
    sintoma_norm = " ".join(sintoma_raw.split())
    return MAPA_SINTOMAS.get(sintoma_norm, sintoma_raw)

def obtener_traduccion_enfermedad(enfermedad_raw):
    enfermedad_clean = enfermedad_raw.strip()
    if enfermedad_clean in MAPA_ENFERMEDADES:
        return MAPA_ENFERMEDADES[enfermedad_clean]
    if enfermedad_raw in MAPA_ENFERMEDADES:
        return MAPA_ENFERMEDADES[enfermedad_raw]
    enfermedad_norm = " ".join(enfermedad_raw.split())
    return MAPA_ENFERMEDADES.get(enfermedad_norm, enfermedad_raw)

# 5. Cargar Modelo y Lista de Síntomas (.pkl)
@st.cache_resource
def cargar_recursos():
    modelo = joblib.load('modelo_definitivo.pkl')
    sintomas = joblib.load('lista_sintomas.pkl')
    return modelo, sintomas

try:
    modelo_definitivo, lista_sintomas = cargar_recursos()
    st.sidebar.success(f"✅ Modelo cargado ({len(lista_sintomas)} síntomas)")
except Exception as e:
    st.error(f"❌ Error al cargar los archivos .pkl: {e}")
    st.stop()

# Generar opciones traducidas y mapa inverso
inverso_sintomas = {obtener_traduccion_sintoma(s): s for s in lista_sintomas}
opciones_espanol = sorted(list(inverso_sintomas.keys()))

# Sidebar: Selección Dinámica de Síntomas
st.sidebar.header("📋 Evaluación de Síntomas")
st.sidebar.markdown("Marca los síntomas observados en el paciente:")

sintomas_espanol_sel = st.sidebar.multiselect(
    "Buscar y seleccionar síntomas:",
    options=opciones_espanol,
    default=[]
)

# Recuperar los nombres de columna originales (en inglés) para el modelo
sintomas_presentes_ingles = [inverso_sintomas[s] for s in sintomas_espanol_sel]

# 6. Construir vector de entrada (1 x N_sintomas)
input_vector = pd.DataFrame(0, index=[0], columns=lista_sintomas)

for sintoma_en in sintomas_presentes_ingles:
    if sintoma_en in input_vector.columns:
        input_vector[sintoma_en] = 1

# 7. Panel Principal - Inferencia y Gráficos
col1, col2 = st.columns([1, 1.2])

if len(sintomas_presentes_ingles) > 0:
    probabilidades = modelo_definitivo.predict_proba(input_vector)[0]
    clases = modelo_definitivo.classes_

    # Crear DataFrame ordenado y traducir diagnósticos con normalización
    df_prob = pd.DataFrame({
        'Enfermedad_EN': clases,
        'Probabilidad': probabilidades * 100
    })
    
    df_prob['Enfermedad'] = df_prob['Enfermedad_EN'].apply(obtener_traduccion_enfermedad)
    df_prob = df_prob.sort_values(by='Probabilidad', ascending=False).reset_index(drop=True)

    top_diagnostico = df_prob.iloc[0]['Enfermedad']
    top_prob = df_prob.iloc[0]['Probabilidad']

    with col1:
        st.subheader("📊 Análisis Diagnóstico")
        
        st.metric(
            label="Diagnóstico Más Probable", 
            value=top_diagnostico, 
            delta=f"{top_prob:.1f}% Probabilidad"
        )

        st.markdown("**Top 5 Diagnósticos Posibles:**")
        df_top5 = df_prob[['Enfermedad', 'Probabilidad']].head(5).copy()
        df_top5['Probabilidad'] = df_top5['Probabilidad'].map("{:.2f}%".format)
        st.dataframe(df_top5, use_container_width=True)

    with col2:
        st.subheader("📈 Distribución de Probabilidades (Top 5)")
        
        df_top5_graph = df_prob.head(5).sort_values(by='Probabilidad', ascending=True)
        
        fig = px.bar(
            df_top5_graph,
            x='Probabilidad',
            y='Enfermedad',
            orientation='h',
            text=df_top5_graph['Probabilidad'].apply(lambda x: f'{x:.1f}%'),
            title="Predicción según el perfil de síntomas"
        )
        
        fig.update_traces(
            marker_color='#8cd8f3',
            textposition='outside',
            textfont=dict(size=16, color='white')
        )
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=16),
            title_font=dict(size=18, color='white'),
            xaxis=dict(
                title="Probabilidad (%)",
                range=[0, 110],
                showgrid=False,
                fixedrange=True,
                tickfont=dict(size=14)
            ),
            yaxis=dict(
                title="",
                fixedrange=True,
                tickfont=dict(size=15, color='white')
            ),
            height=380,
            margin=dict(l=10, r=20, t=40, b=10)
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

else:
    with col1:
        st.info("👈 Por favor, selecciona al menos un síntoma en el panel de la izquierda para generar la predicción.")
    with col2:
        st.subheader("📈 Distribución de Probabilidades")
        st.caption("Esperando entrada de síntomas...")