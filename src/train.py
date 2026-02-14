import os
os.environ["MPLBACKEND"] = "Agg" 

import matplotlib
matplotlib.use("Agg")

import sys
import gc
import mlflow
import pandas as pd
import joblib
import json

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
    credit_card_balance,
    clean_feature_names
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


SUBMISSION_FILE_NAME = "submission_kernel02.csv"

EXPERIMENT = "Chap8_Models_Comparison"


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

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
        
        
    # Sélection de valeur true et false
    df_selection_true = df[df['TARGET'] == 1].head(1)
    df_selection_false = df[df['TARGET'] == 0].head(1)
    
    # Export en JSON pour récupérer des valeurs pour tester API
    df_selection_true.to_json(
        'donnees_test_true.json', 
        orient='records', 
        indent=3, 
        force_ascii=False
    )

    print(f"Export (true) terminé : {len(df_selection_true)} lignes enregistrées.")
    
    df_selection_false.to_json(
        'donnees_test_false.json', 
        orient='records', 
        indent=3, 
        force_ascii=False
    )
    
    print(f"Export (False) terminé : {len(df_selection_false)} lignes enregistrées.")
        
    sys.exit(0)
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
        results = kfold_lightgbm(
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

        # Log métriques
        mlflow.log_metric(f"{model_type}_auc_fold",
                    roc_auc_score(y, probas))
        mlflow.log_metric(f"{model_type}_recall_fold",
                            recall_score(y, preds))
        mlflow.log_metric(f"{model_type}_f1_fold",
                            f1_score(y, preds))
        mlflow.log_metric(f"{model_type}_business_cost",
                            business_cost(y, preds))
        

        # Seuil optimal issu du CV
        best_threshold = float(best_run.data.params.get("best_threshold", 0.5))
        mlflow.log_param("best_threshold", best_threshold)
        
        if hasattr(final_model, "feature_importances_"):
            fi = pd.Series(
                final_model.feature_importances_,
                index=feats
            ).sort_values(ascending=False)

            fi_path = OUTPUT_DIR / "final_feature_importance.csv"
            fi.to_csv(fi_path)

            mlflow.log_artifact(fi_path)
   
        
        # Log du modèle dans le run
        mlflow.sklearn.log_model(
            sk_model=final_model,
            artifact_path="model"
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
    # Sauvegardes (HORS modèle)
    # ==========================

    if not debug:
        submission = pd.DataFrame({
            "SK_ID_CURR": results["test_ids"],
            "TARGET": results["sub_preds"]
        })

        submission.to_csv(SUBMISSION_FILE_NAME, index=False)
        mlflow.log_artifact(SUBMISSION_FILE_NAME)

    fi_path = OUTPUT_DIR / "feature_importance.csv"
    results["feature_importance"].to_csv(fi_path, index=False)
    mlflow.log_artifact(str(fi_path))

if __name__ == "__main__":
    # MLFlow en local
    mlflow.set_tracking_uri("http://127.0.0.1:5001")
    mlflow.set_experiment(EXPERIMENT)   
    with mlflow.start_run(run_name="Full_pipeline"):
        with timer("Full model run"):
            main()