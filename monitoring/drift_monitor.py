import numpy as np
import pandas as pd
import json
import os

from pathlib import Path
from scipy.stats import ks_2samp
from datetime import datetime

from elasticsearch import Elasticsearch


# --- CONFIGURATION ---
INDEX_PROD = os.getenv("INDEX_PROD", "predictions")
INDEX_METRICS = os.getenv("INDEX_METRICS", "drift")

# --- ElasticSarch avec
ES_HOST = os.getenv("ES_HOST", "localhost")
es = Elasticsearch(f"http://{ES_HOST}:9200")

BASE_DIR = Path(__file__).resolve().parent.parent
REF_DIR = BASE_DIR / "reference_distributions"

# ------- Features
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

def run_global_monitoring():
    # 1. Récupérer les données de production (Dernières 24h)
    """ query = {
        "size": 2000, 
        "query": {"range": {"timestamp": {"gte": "now-1d/d", "lt": "now/d"}}}
    } """
    
    # 1. Récupération des dernières valeurs dans Elasticsearch
    try:
        query = {
            "size": 2000,
            "sort": [
                {
                    "@timestamp": {
                        "order": "desc"
                    }
                }
            ],
            "_source": [
                "@timestamp",
                "request_body",
                "path",
                "method"
            ],
            "query": {
                "bool": {
                    "must": [
                        {"term": {"path.keyword": "/predict"}},
                        {"term": {"method.keyword": "POST"}}
                    ],
                    "filter": [
                        {
                            "range": {
                                "@timestamp": {
                                    "gte": "now-1d",
                                    "lte": "now"
                                }
                            }
                        }
                    ]
                }
            }
        }
        response = es.search(index=INDEX_PROD, body=query)
        hits = response["hits"]["hits"]
        print(f"[STEP 1] Docs trouvés dans {INDEX_PROD}: {len(hits)}")
    except Exception as e:
        print(f"[ERREUR STEP 1] Connexion ES impossible: {e}")
        return

    print(f"[LEN Hits] : {len(hits)}")      
    if len(hits) < 50:
        print("Échantillon trop faible pour un monitoring significatif.")
        return

    # Conversion des logs en DataFrame pour faciliter la manipulation
    # En ne prennant que api-logs ayant request_body
    df_prod = pd.DataFrame([
        hit["_source"]["request_body"]
        for hit in hits
        if hit["_source"].get("request_body") is not None
    ])
    
    if df_prod.nunique().mean() < 2:
        print("⚠️ Données quasi constantes → monitoring ignoré")
        return

    for feature_name in (features + boolean_features):

        if feature_name not in df_prod.columns:
            continue

        try:
            with open(f"reference_distributions/{feature_name}.json", "r") as f:
                ref_data = json.load(f)
        except FileNotFoundError:
            continue

        current_values = df_prod[feature_name].dropna()

        if feature_name in boolean_features:
            current_values = current_values.astype(float)

        print(f"[CURRENT] {feature_name} size:", len(current_values))

        if len(current_values) < 30:
            continue
        

        # --- CAS NUMÉRIQUE (Test K-S) ---
        if ref_data["type"] == "numeric":
            ref_values = np.array(ref_data["values"])
              
            stat, p_value = ks_2samp(ref_values, current_values)
            drift_detected = (p_value < 0.05) and (stat > 0.1)
            metric_val = p_value
            metric_name = "p_value_ks"

        # --- CAS BOOLÉEN (Comparaison de proportion) ---
        else:
            # On compare le % de "1" (True)
            ref_rate = ref_data["distribution"].get("1.0", ref_data["distribution"].get("1", 0))
            prod_rate = current_values.mean() # Moyenne d'une colonne 0/1 = taux de 1

            # Pour les booléens, on utilise souvent une différence absolue (ex: > 10%)
            diff = abs(ref_rate - prod_rate)
            drift_detected = diff > 0.10 
            metric_val = diff
            metric_name = "abs_diff_rate"

        # 2. Envoyer le score vers Elasticsearch
        log_data = {
            "@timestamp": datetime.now().isoformat(),
            "feature": feature_name,
            "type": ref_data["type"],
            "metric_name": metric_name,
            "statistic": float(stat),
            "value": float(metric_val),
            "status": "CRITICAL" if drift_detected else "OK"
        }
        print(f"LOG_DATA : {INDEX_METRICS}    {log_data}")
        es.index(index=INDEX_METRICS, document=log_data)
        
    print(f"Monitoring terminé à {datetime.now()}")
    
    
if __name__ == "__main__":
    run_global_monitoring()
        