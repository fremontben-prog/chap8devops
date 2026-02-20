import os
os.environ["MPLBACKEND"] = "Agg" 

import matplotlib
matplotlib.use("Agg")

import gc
import mlflow
import pandas as pd
import joblib
import json
import shap

from datetime import datetime
from pathlib import Path

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, recall_score, f1_score

from src.feature import (
    application_train_test, 
    timer, 
    bureau_and_balance, 
    previous_applications, 
    pos_cash, 
    installments_payments, 
    credit_card_balance
)

from src.model import (
    kfold_lightgbm,
    kfold_random_forest,
    kfold_logistic_regression,
    kfold_xgboost,
    business_cost
)

from src.util import (
    get_best_run,
    get_best_model_info
)

from monitoring.drift_monitor import INDEX_PROD

from elasticsearch import Elasticsearch


SUBMISSION_FILE_NAME = "submission_kernel02.csv"

EXPERIMENT = "Chap8_Models_Comparison"

# Pour l'ensemble des documents en sortie
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# --- ElasticSarch avec
ES_HOST = os.getenv("ES_HOST", "localhost")
es = Elasticsearch(f"http://{ES_HOST}:9200")


# ==============================
# Drift
# ==============================
REF_DIR = Path("reference_distributions")
REF_DIR.mkdir(exist_ok=True)

# Dossier où stocker les références pour le drift
os.makedirs("reference_distributions", exist_ok=True)

# Liste complète des features importantes pour le drift
FEATURES = [
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

# Variables booléennes (one-hot) pour le drift
BOOLEAN_FEATURES = [
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

index_prod = INDEX_PROD


# Transformation des types de paramètres car lorsqu'ils sont stockés dans MLFLOW, ils sont en String uniquement
def cast_params(params):
    casted = {}

    for k, v in params.items():

        # ===== Cas None (PRIORITAIRE) =====
        if v is None or v == "None":
            casted[k] = None
            continue

        # ===== Cas spécial XGBoost =====
        if k == "verbosity":
            casted[k] = int(v)
            continue

        if k == "max_bin":
            casted[k] = int(v)
            continue

        # ===== Cast standard =====
        try:
            casted[k] = int(v)
        except (ValueError, TypeError):
            try:
                casted[k] = float(v)
            except (ValueError, TypeError):
                casted[k] = v

    return casted



def main(debug = True):
    num_rows = 10000 if debug else None
    df = application_train_test(num_rows)

    with timer("Process bureau and bureau_balance"):
        bureau = bureau_and_balance(num_rows)
        df = df.join(bureau, how="left", on="SK_ID_CURR")
        del bureau
        gc.collect()

    with timer("Process previous_applications"):
        prev = previous_applications(num_rows)
        df = df.join(prev, how="left", on="SK_ID_CURR")
        del prev
        gc.collect()

    with timer("Process POS-CASH balance"):
        pos = pos_cash(num_rows)
        df = df.join(pos, how="left", on="SK_ID_CURR")
        del pos
        gc.collect()

    with timer("Process installments payments"):
        ins = installments_payments(num_rows)
        df = df.join(ins, how="left", on="SK_ID_CURR")
        del ins
        gc.collect()

    with timer("Process credit card balance"):
        cc = credit_card_balance(num_rows)
        df = df.join(cc, how="left", on="SK_ID_CURR")
        del cc
        gc.collect()
        
    # ===============================
    # Sauvegarde des valeurs traitées pour tests CI
    # ===============================        
    # Sélection de valeurs true et false pour tests rapide
    df_selection_true = df[df['TARGET'] == 1].head(1)
    df_selection_false = df[df['TARGET'] == 0].head(1)
    
    # Sélection de valeurs true et false pour tests de tout le jeu de test
    df_selection_complet_true = df[df['TARGET'] == 1].head(100)
    df_selection_complet_false = df[df['TARGET'] == 0].head(100)
    
    # Export en JSON pour récupérer des valeurs pour tester API rapidement
    output_file = OUTPUT_DIR / 'donnees_test_true.json'
    df_selection_true.to_json(
        output_file, 
        orient='records', 
        indent=3, 
        force_ascii=False
    )

    print(f"Export (true) terminé : {len(df_selection_true)} lignes enregistrées.")
    
    output_file = OUTPUT_DIR / 'donnees_test_false.json'
    df_selection_false.to_json(
        output_file, 
        orient='records', 
        indent=3, 
        force_ascii=False
    )
    
    print(f"Export (False) terminé : {len(df_selection_false)} lignes enregistrées.")
    
    
    # Export en JSON pour récupérer des valeurs pour tester API jeu de tests complet
    output_file = OUTPUT_DIR / 'donnees_test_full_true.json'
    df_selection_complet_true.to_json(
        output_file, 
        orient='records', 
        indent=3, 
        force_ascii=False
    )

    print(f"Export (true) terminé : {len(df_selection_complet_true)} lignes enregistrées.")
    
    output_file = OUTPUT_DIR / 'donnees_test_full_false.json'
    df_selection_complet_false.to_json(
        output_file, 
        orient='records', 
        indent=3, 
        force_ascii=False
    )
    
    print(f"Export (False) terminé : {len(df_selection_complet_false)} lignes enregistrées.")
    
        
    with timer("Run LogisticRegression"):
        lr_results = kfold_logistic_regression(
            df,
            num_folds=5,
            stratified=True
        )
        
        
    with timer("Run RandomForest"):
        rf_results = kfold_random_forest(
            df,
            num_folds=5,
            stratified=True
        )
        
    with timer("Run XGBoost"):
        xgb_results = kfold_xgboost(
            df,
            num_folds=5,
            stratified=True
        )
    with timer("Run LightGBM with kfold"):
        lgbm_results = kfold_lightgbm(
            df,
            num_folds=10,
            stratified=True
        )

    # ==========================
    # Sélection meilleur modèle
    # ========================== 
    best_run, model_type, best_params = get_best_model_info(
        experiment_name=EXPERIMENT
    )

    print("Best run:", best_run.info.run_id)
    print("Best model type:", model_type)
    print("Best business cost:", best_run.data.metrics["business_cost"])
 
    best_threshold = best_run.data.params.get("best_threshold", 0.5)
    best_threshold = float(best_threshold)

    mlflow.log_param("best_threshold", best_threshold)


    # Entrainement du meilleur model sur les données de Train
    params = cast_params(best_params)
    
    train_df = df[df["TARGET"].notnull()]
 
    feats = [
        f for f in df.columns
        if f not in ["TARGET", "SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV", "index"]
    ]
       
    X = train_df[feats]
    y = train_df["TARGET"].astype(int)

    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    
    if model_type == "xgboost":
        # Suppression de paramètres provenant de MLFlow et non nécessaires
        XGB_FORBIDDEN_PARAMS = {"best_threshold", "use_label_encoder"}
        params = {
            k: v for k, v in cast_params(best_params).items()
            if k not in XGB_FORBIDDEN_PARAMS
        }

        params["scale_pos_weight"] = scale_pos_weight
        final_model = XGBClassifier(**params, use_label_encoder=False)

    elif model_type == "lightgbm":
        params["scale_pos_weight"] = scale_pos_weight
        final_model = LGBMClassifier(**params)

    elif model_type == "random_forest":
        final_model = RandomForestClassifier(**params)

    elif model_type == "logistic_regression":
        final_model = LogisticRegression(**params, max_iter=1000)

    else:
        raise ValueError(f"Modèle inconnu : {model_type}")


    # ==========================
    # Entrainement et bascule du meilleur modèle dans la registery
    # ==========================
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(
        run_name="final_model_training",
        nested=True):
       
        mlflow.set_tag("model_type", model_type)
        mlflow.log_params(params)

        final_model.fit(X, y)
        
       # Prédictions avec le seuil optimal
        probas = final_model.predict_proba(X)[:, 1]
        preds = (probas >= best_threshold).astype(int)
        
        # Métriques
        auc = roc_auc_score(y, probas)
        recall = recall_score(y, preds)
        f1 = f1_score(y, preds)
        business = business_cost(y, preds)
        
        # Log métriques
        mlflow.log_metric(f"{model_type}_auc_fold", auc)
        mlflow.log_metric(f"{model_type}_recall_fold", recall)
        mlflow.log_metric(f"{model_type}_f1_fold", f1)
        mlflow.log_metric(f"{model_type}_business_cost", business)
        

        # Seuil optimal issu du CV
        best_threshold = float(best_run.data.params.get("best_threshold", 0.5))
        mlflow.log_param("best_threshold", best_threshold)
        
        
        
        if hasattr(final_model, "feature_importances_"):
            fi = pd.Series(
                final_model.feature_importances_,
                index=feats
            ).sort_values(ascending=False)

            fi_path = OUTPUT_DIR / f"final_{model_type}_feature_importance.csv"
            print(f"Feature importance : {fi_path}")
            fi.to_csv(fi_path)

            mlflow.log_artifact(fi_path)
   
        
        # Log du modèle dans le run
        mlflow.sklearn.log_model(
            sk_model=final_model,
            name="model"
        )
        
        # Sauvegarde pour être utilisé par la CI 
        # Sauvegarde du modèle et threshold
        fi_path = OUTPUT_DIR / "final_model.pkl"
        joblib.dump(final_model, fi_path)

        fi_path = OUTPUT_DIR / "best_threshold.json"
        with open(fi_path, "w") as f:
            json.dump({"best_threshold": best_threshold}, f)

        print("Model and threshold saved successfully.")

        # URI exact du modèle
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"

        # Enregistrement dans la registry
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name="CreditDefaultModel"
        )

    
    # ==========================
    # Sauvegarde des métriques dans ElasticSearch 
    # pour le dashboard
    # ==========================
    print(f"@timestamp :{datetime.now().isoformat()}")
    print(f"f1 : {float(f1)}")
    print(f"recall : {float(recall)}")
    print(f"auc : {float(auc)}")
          
    es.index(index="model-performance", document={
        "@timestamp": datetime.now().isoformat(),
        "model_version": "v1.0",
        "f1": float(f1),
        "recall": float(recall),
        "auc": float(auc)
    })
        
    # ==========================
    # Promotion en Staging
    # ==========================
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(
        run_name="final_model_staging",
        nested=True):
        client = mlflow.MlflowClient()
        client.set_registered_model_alias(
            name="CreditDefaultModel",
            alias="staging",
            version=registered_model.version
        )


    # ==========================
    # Sauvegarde des features
    # ==========================
      
    if model_type == "xgboost":
        # Sauvegarde des features importances
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X)

        shap_importance = pd.DataFrame({
            "feature": X.columns,
            "importance": abs(shap_values).mean(axis=0)
        }).sort_values("importance", ascending=False)

        shap_importance_path = OUTPUT_DIR / f"final_{model_type}_shap_importance.csv"
        shap_importance.to_csv(shap_importance_path, index=False)

        mlflow.log_artifact(shap_importance_path)
        
        
    # ==========================
    # Drift
    # ==========================
    def save_numeric_reference(series, feature_name):
        missing_rate = float(series.isna().mean())
        series_clean = series.dropna()

        sample_data = series_clean.sample(
            min(len(series_clean), 10000),
            random_state=42
        ).tolist()

        ref = {
            "type": "numeric",
            "values": sample_data,
            "mean": float(series_clean.mean()),
            "std": float(series_clean.std()),
            "missing_rate": missing_rate
        }

        with open(REF_DIR / f"{feature_name}.json", "w") as f:
            json.dump(ref, f)

    def save_boolean_reference(series, feature_name):
        # normalize=True donne directement les proportions (ex: {True: 0.8, False: 0.2})
        distribution = series.value_counts(normalize=True).to_dict()
        
        # On s'assure que les clés sont des strings pour le JSON
        distribution = {str(k): v for k, v in distribution.items()}

        ref = {
            "type": "boolean",
            "distribution": distribution,
            "missing_rate": float(series.isna().mean())
        }

        with open(REF_DIR / f"{feature_name}.json", "w") as f:
            json.dump(ref, f)

    # Application
    for feature in FEATURES:
        if feature in BOOLEAN_FEATURES:
            save_boolean_reference(X[feature], feature)
        else:
            save_numeric_reference(X[feature], feature)

    print("Références sauvegardées avec succès.")
    

"""     fi_path = OUTPUT_DIR / f"{model_type}_feature_importance.csv"
    print(f"Feature importance : {fi_path}")
    results["feature_importance"].to_csv(fi_path, index=False)
    mlflow.log_artifact(str(fi_path)) """

if __name__ == "__main__":
    # MLFlow en local
    mlflow.set_tracking_uri("http://127.0.0.1:5001")
    mlflow.set_experiment(EXPERIMENT)   
    with mlflow.start_run(run_name="Full_pipeline"):
        with timer("Full model run"):
            main()