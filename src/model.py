# Bibliothèques de base
import numpy as np
import pandas as pd
import lightgbm as lgb
import mlflow
import mlflow.lightgbm


from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


from sklearn.metrics import roc_auc_score, recall_score, f1_score

from sklearn.model_selection import KFold, StratifiedKFold


import matplotlib.pyplot as plt
from src.feature import clean_feature_names

import optuna

EXPERIMENT = "Chap8_Models_Comparison"

from pathlib import Path
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def optimize_lgbm_with_optuna(train_df, feats, folds, n_trials=30):
    def objective(trial):
        # Calcul du déséquilibre des classes via le paramètre scale_pos_weight de LightGBM comme le ratio entre classes majoritaire et minoritaire.
        y = train_df["TARGET"].values
        scale_pos_weight = (y == 0).sum() / (y == 1).sum()

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 500, 3000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "num_leaves": trial.suggest_int("num_leaves", 16, 64),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "class_weight": "balanced",
            "scale_pos_weight": scale_pos_weight,
            "objective": "binary",
            "n_jobs": 4,
            "random_state": 42
        }

        model = LGBMClassifier(**params)

        fit_params = {
            "eval_metric": "auc",
            "callbacks": [lgb.early_stopping(100)]
        }

        oof_preds, _, _ = run_cv(
            model=model,
            train_df=train_df,
            test_df=train_df,  
            feats=feats,
            folds=folds,
            model_name="optuna_lgbm",
            fit_params=fit_params,
            log_model_fn=None,
            collect_importance=False
        )

        y_true = train_df["TARGET"].values
        result = find_best_threshold(y_true, oof_preds)

        return result["best_cost"]  # objectif métier

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    return study


def plot_cost_threshold(thresholds, costs, output_path="cost_vs_threshold.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, costs)
    plt.xlabel("Seuil de décision")
    plt.ylabel("Coût métier")
    plt.title("Coût métier en fonction du seuil")
    plt.grid(True)

    plt.tight_layout()
    output_path = OUTPUT_DIR / output_path
    plt.savefig(output_path)
    plt.close()

    return output_path

# Fonction coût métier
def business_cost(y_true, y_pred, cost_fn=5, cost_fp=1):
    """
    y_true : vrais labels (0/1)
    y_pred : prédictions binaires (0/1)
    """
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    return cost_fn * fn + cost_fp * fp

# Tester les seuils de 0.1 à 0.9
def find_best_threshold(y_true, y_proba):
    thresholds = np.arange(0.1, 0.9, 0.01)
    costs = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        cost = business_cost(y_true, y_pred)
        costs.append(cost)

    best_idx = np.argmin(costs)

    return {
        "best_threshold": thresholds[best_idx],
        "best_cost": costs[best_idx],
        "thresholds": thresholds,
        "costs": costs
    }

# Fonction exécution de de cross validate pour tous les modèles
def run_cv(
    model,
    train_df,
    test_df,
    feats,
    folds,
    model_name,
    fit_params=None,
    log_model_fn=None,
    collect_importance=False
):
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()

    for n_fold, (train_idx, valid_idx) in enumerate(
        folds.split(train_df[feats], train_df["TARGET"])
    ):
        train_x = train_df[feats].iloc[train_idx]
        train_y = train_df["TARGET"].iloc[train_idx]
        valid_x = train_df[feats].iloc[valid_idx]
        valid_y = train_df["TARGET"].iloc[valid_idx]

          
        if fit_params:
            model.fit(
                train_x,
                train_y,
                eval_set=[(valid_x, valid_y)],
                **fit_params
            )
        else:
            model.fit(train_x, train_y)

            
        
        if log_model_fn:
            log_model_fn(model, n_fold)

        probas = model.predict_proba(valid_x)[:, 1]
        oof_preds[valid_idx] = probas
        sub_preds += model.predict_proba(test_df[feats])[:, 1] / folds.n_splits

        preds = (probas > 0.5).astype(int)

        mlflow.log_metric(f"{model_name}_auc_fold_{n_fold+1}",
                          roc_auc_score(valid_y, probas))
        mlflow.log_metric(f"{model_name}_recall_fold_{n_fold+1}",
                          recall_score(valid_y, preds))
        mlflow.log_metric(f"{model_name}_f1_fold_{n_fold+1}",
                          f1_score(valid_y, preds))
        mlflow.log_metric(f"{model_name}_business_cost_fold_{n_fold+1}",
                          business_cost(valid_y, preds))

        if collect_importance and hasattr(model, "feature_importances_"):
            fold_importance = pd.DataFrame({
                "feature": feats,
                "importance": model.feature_importances_,
                "fold": n_fold + 1
            })
            feature_importance_df = pd.concat(
                [feature_importance_df, fold_importance]
            )

    mlflow.log_metric(
        f"{model_name}_auc_cv",
        roc_auc_score(train_df["TARGET"], oof_preds)
    )

    return oof_preds, sub_preds, feature_importance_df



# Modèle : LightGBM GBDT with KFold or Stratified KFold
def kfold_lightgbm(df, num_folds, stratified=True):
    mlflow.set_experiment(EXPERIMENT)
    
    # ==========================
    # Préparation des données
    # ==========================
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df = clean_feature_names(df)

    train_df = df[df["TARGET"].notnull()]
    test_df = df[df["TARGET"].isnull()]

    feats = [
        f for f in train_df.columns
        if f not in ["TARGET", "SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV", "index"]
    ]

    folds = StratifiedKFold(
        n_splits=num_folds,
        shuffle=True,
        random_state=1001
    )


    # ======================================================
    # RUN OPTUNA — Recherche hyperparamètres
    # ======================================================
    with mlflow.start_run(
        run_name="LightGBM_Optuna_Search",
        nested=True
    ):
        print(">>> Optuna optimization started")

        study = optimize_lgbm_with_optuna(
            train_df=train_df,
            feats=feats,
            folds=folds,
            n_trials=30
        )

        best_params = study.best_params

        mlflow.log_params(best_params)
        mlflow.log_metric("optuna_best_business_cost", study.best_value)

    # ======================================================
    # RUN MODÈLE FINAL — Entraînement + CV
    # ======================================================
    with mlflow.start_run(
        run_name="LightGBM_Final_Model",
        nested=True
    ):
        print(">>> LightGBM final model training")
        model_name="lightgbm"
        
        # Calcul du déséquilibre des classes via le paramètre scale_pos_weight de LightGBM comme le ratio entre classes majoritaire et minoritaire.
        y = train_df["TARGET"].values
        scale_pos_weight = (y == 0).sum() / (y == 1).sum()

        model = LGBMClassifier(
            **best_params,
            objective="binary",
            n_jobs=4,
            random_state=42
        )

        mlflow.log_params(model.get_params())
        mlflow.set_tag("model_type", model_name)

        fit_params = {
            "eval_metric": "auc",
            "callbacks": [lgb.early_stopping(200)]
        }

        def log_lgbm_model(m, fold):
            mlflow.lightgbm.log_model(
                m,
                name=f"{model_name}_fold_{fold+1}"
            )

        oof_preds, sub_preds, feature_importance = run_cv(
            model=model,
            train_df=train_df,
            test_df=test_df,
            feats=feats,
            folds=folds,
            model_name=model_name,
            fit_params=fit_params,
            log_model_fn=log_lgbm_model,
            collect_importance=True
        )

        # =====================================
        # Optimisation du seuil métier
        # =====================================
        y_true = train_df["TARGET"].values

        result = find_best_threshold(y_true, oof_preds)

        mlflow.log_metric("final_business_cost", result["best_cost"])
        mlflow.log_param("best_threshold", result["best_threshold"])

        plot_path = plot_cost_threshold(
            result["thresholds"],
            result["costs"],
            output_path="lightgbm_cost_vs_threshold.png"
        )

        mlflow.log_artifact(plot_path)

    return {
        "oof_preds": oof_preds,
        "sub_preds": sub_preds,
        "test_ids": test_df["SK_ID_CURR"].values,
        "feature_importance": feature_importance
    }

    
    
    
# Modèle : Random forest     
def kfold_random_forest(df, num_folds, stratified=True):
    mlflow.set_experiment(EXPERIMENT)
    
    with mlflow.start_run(run_name="RandomForest_home_credit", nested=True):
        print(">>> RandomForest run started")
        model_name="random_forest"
        
        # Conversion des colonnes object
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        df = clean_feature_names(df)

        train_df = df[df["TARGET"].notnull()]
        test_df = df[df["TARGET"].isnull()]

        feats = [f for f in train_df.columns if f not in
                ["TARGET", "SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV", "index"]]

        folds = StratifiedKFold(num_folds, shuffle=True, random_state=1001)

        model = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=20,
            class_weight="balanced",
            n_jobs=4,
            random_state=42
        )

        mlflow.log_params(model.get_params())
        mlflow.set_tag("model_type", model_name)

        def log_rf_model(m, fold):
            mlflow.sklearn.log_model(
                m,
                name=f"{model_name}_fold_{fold+1}"
            )

        
        oof_preds, sub_preds, feature_importance = run_cv(
            model=model,
            train_df=train_df,
            test_df=test_df,
            feats=feats,
            folds=folds,
            model_name=model_name,
            log_model_fn=log_rf_model,
            collect_importance=True
        )
        
    
        # ==========================
        # Optimisation seuil métier
        # ==========================

        y_true = train_df["TARGET"].values

        result = find_best_threshold(y_true, oof_preds)

        mlflow.log_metric("business_cost", result["best_cost"])
        mlflow.log_param("best_threshold", result["best_threshold"])

        plot_path = plot_cost_threshold(
            result["thresholds"],
            result["costs"],
            output_path=f"{model_name}_cost_vs_threshold.png"
        )

        mlflow.log_artifact(plot_path)


    return {
        "oof_preds": oof_preds,
        "sub_preds": sub_preds,
        "test_ids": test_df["SK_ID_CURR"].values,
        "feature_importance": feature_importance
    }
    
    
# Modèle : Logistique régression    
def kfold_logistic_regression(df, num_folds, stratified=True):
    mlflow.set_experiment(EXPERIMENT)
    model_name="logistic_regression"
    
    with mlflow.start_run(run_name="LogisticRegression_home_credit", nested=True):
        mlflow.set_tag("model_type", model_name)
        print(">>> LogisticRegression run started")

        # Conversion des colonnes object
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        df = clean_feature_names(df)

        train_df = df[df["TARGET"].notnull()]
        test_df = df[df["TARGET"].isnull()]

        feats = [f for f in train_df.columns if f not in
                    ["TARGET", "SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV", "index"]]

        folds = StratifiedKFold(num_folds, shuffle=True, random_state=1001)

        # Encapsulation de LogistiRegression pour gérer les Nan qui étaient tolérés
        model = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                solver="liblinear",
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            ))
        ])

        mlflow.log_params({
            "model": "LogisticRegression",
            "imputer": "median",
            "scaler": "standard",
            "class_weight": "balanced"
        })


        def log_lr_model(m, fold):
            mlflow.sklearn.log_model(
                m,
                name=f"{model_name}_fold_{fold+1}"
            )

        oof_preds, sub_preds, _ = run_cv(
            model=model,
            train_df=train_df,
            test_df=test_df,
            feats=feats,
            folds=folds,
            model_name=model_name,
            log_model_fn=log_lr_model,
            collect_importance=False
        )
        
        y_true = train_df["TARGET"].values
        result = find_best_threshold(y_true, oof_preds)

        mlflow.log_metric("business_cost", result["best_cost"])
        mlflow.log_param("best_threshold", result["best_threshold"])
        
        plot_path = plot_cost_threshold(
            result["thresholds"],
            result["costs"],
            output_path=f"{model_name}_cost_vs_threshold.png"
        )
        mlflow.log_artifact(plot_path)

    return {
        "oof_preds": oof_preds,
        "sub_preds": sub_preds,
        "test_ids": test_df["SK_ID_CURR"].values
    }


# Modèle : xgboost
def kfold_xgboost(df, num_folds, stratified=True):
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name="XGBoost_home_credit", nested=True):
        print(">>> XGBoost run started")
        model_name="xgboost"

        # Conversion des colonnes object
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = clean_feature_names(df)

        train_df = df[df["TARGET"].notnull()]
        test_df = df[df["TARGET"].isnull()]

        feats = [
            f for f in train_df.columns
            if f not in ["TARGET", "SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV", "index"]
        ]
        
        # Calcul du déséquilibre de classe
        y = train_df["TARGET"].values
        scale_pos_weight = (y == 0).sum() / (y == 1).sum()


        folds = StratifiedKFold(
            n_splits=num_folds,
            shuffle=True,
            random_state=1001
        )

        model = XGBClassifier(
            n_estimators=2000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            n_jobs=4,
            random_state=42
        )

        mlflow.log_params(model.get_params())
        mlflow.set_tag("model_type", "xgboost")

        fit_params = {
            "verbose": False
        }
    
        def log_xgb_model(m, fold):
            mlflow.xgboost.log_model(
                m,
                name=f"{model_name}_fold_{fold+1}"
            )

        oof_preds, sub_preds, feature_importance = run_cv(
            model=model,
            train_df=train_df,
            test_df=test_df,
            feats=feats,
            folds=folds,
            model_name=model_name,
            fit_params=fit_params,
            log_model_fn=log_xgb_model,
            collect_importance=True
        )
        
        y_true = train_df["TARGET"].values
        result = find_best_threshold(y_true, oof_preds)

        mlflow.log_metric("business_cost", result["best_cost"])
        mlflow.log_param("best_threshold", result["best_threshold"])

        plot_path = plot_cost_threshold(
            result["thresholds"],
            result["costs"],
            output_path=f"{model_name}_cost_vs_threshold.png"
        )

        mlflow.log_artifact(plot_path)


    return {
        "oof_preds": oof_preds,
        "sub_preds": sub_preds,
        "test_ids": test_df["SK_ID_CURR"].values,
        "feature_importance": feature_importance
    }
