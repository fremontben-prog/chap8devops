import mlflow
from mlflow.tracking import MlflowClient

def get_best_run(experiment_name, metric_name="business_cost"):
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric_name} ASC"],
        max_results=1
    )

    return runs[0]

import mlflow


def get_best_model_info(experiment_name, metric="business_cost"):
    """
    Retourne le type de modèle et les hyperparamètres
    du meilleur run selon une métrique métier.
    """
    client = mlflow.MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment {experiment_name} introuvable")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} ASC"],  # minimisation du coût
        max_results=1
    )

    best_run = runs[0]

    model_type = best_run.data.tags.get("model_type")
    params = best_run.data.params
    
    if model_type is None:
        raise ValueError("Tag 'model_type' manquant dans le run gagnant")

    return best_run, model_type, params

