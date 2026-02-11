---
title: 
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_file: 
pinned: false
---
# Prédiction de l’attrition des employés (ESN)

## Contexte et objectif

---

## Architecture globale
├── data
│   ├── raw/                # Données sources
│   └── processed/          # Données nettoyées et préparées
├── notebooks/              # EDA, feature engineering, expérimentation
├── src/
│   └── chap6mlflow/
│       ├── api/             # API FastAPI
│       ├── models/          # Modèles ML entraînés
│       ├── preprocessing/   # Pipelines de transformation
│       └── utils/           # Fonctions utilitaires
├── tests/
│   ├── unitaires/
│   ├── fonctionnels/
│   └── api/
├── docs/                    # Documentation MkDocs / Sphinx
├── README.md
└── pyproject.toml

## Technologies utilisées
* Python 3.12
* 

## Description du modèle de Machine Learning
### Type de problème

Classification binaire :
0 = client solvable, 1 = client non solvable

### Données utilisées

* Informations contractuelles
* Ancienneté
* Augmentation salariale
* Performance
* Sondage
* SI RH

### Pipeline ML
1. Nettoyage des données
2. Encodage des variables catégorielles
3. Normalisation des variables numériques
4. Entraînement du modèle
5. Évaluation
6. Sérialisation et déploiement

### Performances
| Modèle | Précision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| Random Forest | 0.35 | 0.39 | 0.37 |

Les métriques sont recalculées à chaque réentraînement.

## Getting Started
### Prérequis

* Python 3.12
* Conda (recommandé)

## Installation
### initiales
pip install -e . # A partir des éléments de pyproject.toml
conda activate chap6mlflow
conda install -n chap6mlflow ipykernel --update-deps --force-reinstall
### A partir de GitLab
1. Cloner le projet
' git clone https://github.com/fremontben-prog/Chap5Git.git
' cd Chap5Git

2. Créer l’environnement virtuel
' conda create --name chap6mlflowgit python=3.10
' conda activate chap6mlflow

3. Installer les dépendances
' pip install -e .

4. Vérifier la version Python
' python --version

## Lancer l’API
' cd src
' uvicorn chap5git.api.api_model:app --reload


* API : http://127.0.0.1:8000
* Swagger UI : http://127.0.0.1:8000/docs
* OpenAPI JSON : http://127.0.0.1:8000/openapi.json

## Documentation de l’API

L’API est entièrement documentée via Swagger / OpenAPI, intégré nativement à FastAPI.

Exemple d’appel
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "id_employee": 1234,
    "nombre_participation_pee": 2,
    "nb_formations_suivies": 0,
    "distance_domicile_travail": 50,
    "niveau_education": 3,
    "annees_depuis_la_derniere_promotion": 2,
    "frequence_deplacement": "OCCASIONNEL",
    "satisfaction_employee_environnement": 1,
    "note_evaluation_precedente": 1,
    "niveau_hierarchique_poste": 2,
    "satisfaction_employee_nature_travail": 1,
    "satisfaction_employee_equipe": 1,
    "satisfaction_employee_equilibre_pro_perso": 1,
    "note_evaluation_actuelle": 1,
    "augmentation_salaire_precedente": 1,
    "nombre_experiences_precedentes": 4,
    "annee_experience_totale": 10,
    "annees_dans_l_entreprise": 2,
    "annees_dans_le_poste_actuel": 2,
    "genre": "M",
    "poste": "CONSULTANT",
    "domaine_etude": "INFRA & CLOUD",
    "departement": "CONSULTING",
    "statut_marital": "MARIÉ(E)",
    "delta_note_evaluation": 0
  }'

Réponse
{
  "prediction": 1,
  "probability": 0.523
}

### Exécuter les tests
* Lancer tous les tests
' pytest

* Couverture des tests
' htmlcov/index.html


### Maintenance et mise à jour du modèle
* Protocole de mise à jour
* Collecte de nouvelles données RH
* Analyse de dérive des données
* Réentraînement du modèle
* Évaluation comparative
* Versioning du modèle
* Déploiement contrôlé

### Documentation technique

La documentation complète est disponible via :
* MkDocs : documentation utilisateur
' mkdocs serve -a 127.0.0.1:8001
' Documentation    → http://127.0.0.1:8001

## Contribution

Les contributions sont les bienvenues.

### Règles générales

* Revue de code obligatoire
* Workflow Git
* git checkout develop
* git push

### Conventions

Messages de commit : Conventional Commits feat(scope): description fix(scope): description
Branches : develop/*, master/*

Commits : Conventional Commits

## Auteur

B. Frémont

## Licence

Ce projet est sous licence GNU GPL v3