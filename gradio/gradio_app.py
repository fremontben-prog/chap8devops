import os
import requests
import logging
import random
import gradio as gr

# ===========================
# CONFIG LOGGING
# ===========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===========================
# FEATURES
# ===========================
features = [
    "EXT_SOURCE_3","EXT_SOURCE_2","DAYS_ID_PUBLISH","PAYMENT_RATE",
    "AMT_GOODS_PRICE","EXT_SOURCE_1","DAYS_REGISTRATION",
    "DAYS_EMPLOYED_PERC","AMT_ANNUITY","DAYS_BIRTH",
    "DAYS_LAST_PHONE_CHANGE","REGION_POPULATION_RELATIVE",
    "DAYS_EMPLOYED","ANNUITY_INCOME_PERC","INCOME_PER_PERSON",
    "INCOME_CREDIT_PERC","AMT_CREDIT","HOUR_APPR_PROCESS_START",
    "OWN_CAR_AGE","AMT_INCOME_TOTAL","AMT_REQ_CREDIT_BUREAU_YEAR",
    "LIVINGAREA_MODE","APARTMENTS_AVG","DEF_30_CNT_SOCIAL_CIRCLE",
    "OBS_60_CNT_SOCIAL_CIRCLE","REGION_RATING_CLIENT_W_CITY",
    "OBS_30_CNT_SOCIAL_CIRCLE","LANDAREA_MODE","TOTALAREA_MODE",
    "LANDAREA_AVG","YEARS_BEGINEXPLUATATION_AVG",
    "YEARS_BEGINEXPLUATATION_MODE","APARTMENTS_MODE",
    "CNT_CHILDREN","BASEMENTAREA_MODE","LIVINGAREA_MEDI",
    "LIVINGAREA_AVG","FLOORSMIN_AVG","REG_CITY_NOT_WORK_CITY",
    "YEARS_BEGINEXPLUATATION_MEDI","AMT_REQ_CREDIT_BUREAU_QRT",
    "FLAG_PHONE","APARTMENTS_MEDI","COMMONAREA_MODE",
    "LIVE_CITY_NOT_WORK_CITY","BASEMENTAREA_AVG"
]

boolean_features = [
    "NAME_EDUCATION_TYPE_Higher education",
    "WEEKDAY_APPR_PROCESS_START_TUESDAY",
    "OCCUPATION_TYPE_Laborers",
    "NAME_HOUSING_TYPE_House / apartment",
    "ORGANIZATION_TYPE_Self-employed",
    "WEEKDAY_APPR_PROCESS_START_FRIDAY",
    "NAME_EDUCATION_TYPE_Secondary / secondary special",
    "NAME_INCOME_TYPE_Commercial associate",
    "NAME_FAMILY_STATUS_Married",
    "ORGANIZATION_TYPE_Other",
    "WALLSMATERIAL_MODE_Panel",
    "NAME_CONTRACT_TYPE_Cash loans"
]

# ===========================
# URL API
# ===========================
API_URL = os.getenv("API_URL", "http://api:8000/predict")

# ===========================
# Valeurs de références 
# ===========================
REFERENCE_VALUES = {
    "EXT_SOURCE_3": 0.139, "EXT_SOURCE_2": 0.263, "DAYS_ID_PUBLISH": -2120,
    "PAYMENT_RATE": 0.061, "AMT_GOODS_PRICE": 351000.0, "EXT_SOURCE_1": 0.083,
    "DAYS_REGISTRATION": -3648, "DAYS_EMPLOYED_PERC": 0.067, "AMT_ANNUITY": 24700.5,
    "DAYS_BIRTH": -9461, "DAYS_LAST_PHONE_CHANGE": -1134, "REGION_POPULATION_RELATIVE": 0.0188,
    "DAYS_EMPLOYED": -637, "ANNUITY_INCOME_PERC": 0.122, "INCOME_PER_PERSON": 202500.0,
    "INCOME_CREDIT_PERC": 0.498, "AMT_CREDIT": 406597.5, "HOUR_APPR_PROCESS_START": 10,
    "OWN_CAR_AGE": None, "AMT_INCOME_TOTAL": 202500.0,
    "AMT_REQ_CREDIT_BUREAU_YEAR": 1.0, "LIVINGAREA_MODE": 0.0198, "APARTMENTS_AVG": 0.0247,
    "DEF_30_CNT_SOCIAL_CIRCLE": 2, "OBS_60_CNT_SOCIAL_CIRCLE": 2, "REGION_RATING_CLIENT_W_CITY": 2,
    "OBS_30_CNT_SOCIAL_CIRCLE": 2, "LANDAREA_MODE": 0.0377, "TOTALAREA_MODE": 0.0149,
    "LANDAREA_AVG": 0.0369, "YEARS_BEGINEXPLUATATION_AVG": 0.9722, "YEARS_BEGINEXPLUATATION_MODE": 0.9722,
    "APARTMENTS_MODE": 0.0252, "CNT_CHILDREN": 0, "BASEMENTAREA_MODE": 0.0383,
    "LIVINGAREA_MEDI": 0.0193, "LIVINGAREA_AVG": 0.019, "FLOORSMIN_AVG": 0.125,
    "REG_CITY_NOT_WORK_CITY": 0, "YEARS_BEGINEXPLUATATION_MEDI": 0.9722,
    "AMT_REQ_CREDIT_BUREAU_QRT": 0.0, "FLAG_PHONE": 1, "APARTMENTS_MEDI": 0.025,
    "COMMONAREA_MODE": 0.0144, "LIVE_CITY_NOT_WORK_CITY": 0
}

BOOLEAN_FIELDS = {
    "NAME_EDUCATION_TYPE_Higher education": False,
    "NAME_EDUCATION_TYPE_Secondary / secondary special": True,
    "NAME_INCOME_TYPE_Commercial associate": False,
    "NAME_FAMILY_STATUS_Married": False,
    "NAME_CONTRACT_TYPE_Cash loans": True,
    "WEEKDAY_APPR_PROCESS_START_TUESDAY": False,
    "WEEKDAY_APPR_PROCESS_START_FRIDAY": False,
    "OCCUPATION_TYPE_Laborers": True,
    "NAME_HOUSING_TYPE_House / apartment": True,
    "ORGANIZATION_TYPE_Self-employed": False,
    "ORGANIZATION_TYPE_Other": False,
    "WALLSMATERIAL_MODE_Panel": False
}


# ===========================
# Génération de valeurs à partir des valeurs de référence
# ===========================
def generate_realistic_payload():
    payload = {}
    for f in features:
        v = REFERENCE_VALUES.get(f, 0)
        if v is None:
            payload[f] = None
        elif isinstance(v, float):
            payload[f] = round(random.uniform(0.9*v, 1.1*v), 6)
        else:
            payload[f] = int(v + random.randint(-10,10)) if v != 0 else v
    for b in boolean_features:
        v = BOOLEAN_FIELDS.get(b, False)
        payload[b] = v if random.random() > 0.2 else not v
    # Corrections pour le modèle
    if payload.get("AMT_CREDIT",0) <= 0:
        payload["AMT_CREDIT"] = 1000
    if payload.get("DAYS_BIRTH",0) >= 0:
        payload["DAYS_BIRTH"] = -random.randint(18*365,70*365)
    return payload

# ===========================
# Fonction de prédiction
# ===========================
def gradio_predict(*inputs_values):
    payload = {}
    # Numériques
    num_values = inputs_values[:len(features)]
    for f, v in zip(features, num_values):
        try:
            payload[f] = float(v)
        except Exception:
            payload[f] = 0.0
    # Booléens
    bool_values = inputs_values[len(features):]
    for b, v in zip(boolean_features, bool_values):
        payload[b] = bool(v)
    logger.info(f"PAYLOAD SENT: {payload}")
    try:
        response = requests.post(API_URL, json=payload)
        logger.info(f"STATUS CODE: {response.status_code} | RESPONSE: {response.text}")
        response.raise_for_status()
        data = response.json()
        return data.get("probability"), data.get("prediction")
    except Exception as e:
        logger.error(f"ERROR calling API: {e}")
        return "Erreur API", "Erreur"

# ===========================
# PREDICTION RANDOM
# ===========================
def fill_and_predict():
    payload = generate_realistic_payload()

    # Liste des valeurs dans l'ordre exact des inputs
    input_values = [payload[f] for f in features] + \
                   [payload[b] for b in boolean_features]

    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        prob = data.get("probability")
        pred = data.get("prediction")
    except Exception as e:
        logger.error(f"ERROR calling API: {e}")
        prob = "Erreur"
        pred = "Erreur"

    # On retourne : tous les inputs + les outputs
    return input_values + [prob, pred]

# ===========================
# Interface Gradio
# ===========================
with gr.Blocks() as demo:
    gr.Markdown("# Credit Default Predictor")
    gr.Markdown("Saisissez les valeurs des features pour prédire le risque de défaut")

    # Définition des entrées
    with gr.Row():
        with gr.Column():
            inputs = [gr.Number(label=f) for f in features] + \
                     [gr.Checkbox(label=b) for b in boolean_features]
        
        # Définition des sorties
        with gr.Column():
            outputs = [
                gr.Number(label="Probability"),
                gr.Number(label="Prediction")
            ]

    # Ajout des boutons
    with gr.Row():
        btn_predict = gr.Button("Prédire", variant="primary")
        btn_random = gr.Button("Générer un payload réaliste")

    # Liaison des événements (obligatoirement à l'intérieur du bloc 'with')
    # 1. Action du bouton de prédiction
    btn_predict.click(fn=gradio_predict, inputs=inputs, outputs=outputs)
    
    # 2. Action du bouton "Aléatoire" 
    btn_random.click(
        fn=fill_and_predict,
        inputs=[],
        outputs=inputs + outputs
    )

# ===========================
# LANCEMENT
# ===========================
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, debug=True)