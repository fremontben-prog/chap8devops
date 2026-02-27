# 🚀 Credit Default API – DevOps Project

Projet complet de déploiement d’un modèle de **prédiction de défaut de crédit** avec :

- 🔹 Entraînement + tracking avec MLflow
- 🔹 API REST avec FastAPI
- 🔹 Interface utilisateur avec Gradio
- 🔹 Monitoring via Elasticsearch + Kibana
- 🔹 Conteneurisation Docker
- 🔹 CI/CD avec GitHub Actions
- 🔹 Publication sur Docker Hub

# 📦 Architecture du projet
📦 Projet
┣ 📂 src → Entraînement du modèle
┣ 📂 api → API FastAPI
┣ 📂 gradio_app → Interface utilisateur
┣ 📂 data/raw → Données brutes
┣ 📂 tests → Tests automatisés
┣ 📜 docker-compose.yml → Orchestration
┣ 📜 requirements.txt → Dépendances
┗ 📜 README.md

# 🔬 1️⃣ Lancer MLflow
Avant de démarrer Docker, lancer le serveur MLflow :

```powershell
$env:MLFLOW_SERVER_ALLOWED_HOSTS="*"; mlflow server `
  --backend-store-uri sqlite:///mlflow.db `
  --artifacts-destination "./mlruns" `
  --serve-artifacts `
  --host 0.0.0.0 `
  --port 5001

MLflow sera accessible sur :
http://localhost:5001

🐳 2️⃣ Lancer l environnement Docker
✅ Vérifier Docker Desktop
✅ Vérifier que WSL fonctionne
wsl -l -v
Résultat attendu :
docker-desktop    Running    2

🔨 Construire et lancer les services
docker compose up --build

Arrêter les conteneurs :
docker compose down

Relancer proprement :
docker compose up --build

Voir les logs API :
docker compose logs api

🌍 3️⃣ Accès aux services
Service	URL
API	http://localhost:8000
Swagger	http://localhost:8000/docs
Gradio	http://localhost:7860/gradio
Elasticsearch	http://localhost:9200
Kibana	http://localhost:5601

🔎 4️⃣ Docker Compose
services:

  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.12.0
    container_name: elasticsearch
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - ES_JAVA_OPTS=-Xms512m -Xmx512m
    ports:
      - "9200:9200"
    volumes:
      - esdata:/usr/share/elasticsearch/data

  kibana:
    image: docker.elastic.co/kibana/kibana:8.12.0
    container_name: kibana
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    ports:
      - "5601:5601"
    depends_on:
      - elasticsearch

  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    container_name: api
    ports:
      - "8000:8000"
    environment:
      - ES_HOST=elasticsearch
      - INDEX_PROD=api-logs
      - INDEX_METRICS=drift-metrics
      - MLFLOW_TRACKING_URI=http://host.docker.internal:5001
    volumes:
      - ./reference_distributions:/app/reference_distributions
    depends_on:
      - elasticsearch

  gradio:
    build:
      context: .
      dockerfile: Dockerfile.gradio
    container_name: gradio
    ports:
      - "7860:7860"
    environment:
      - API_URL=http://api:8000/predict
    depends_on:
      - api

volumes:
  esdata:
🔁 5️⃣ CI/CD – GitHub Actions
name: CI/CD - Credit Default API

on:
  push:
    branches: [master, develop]
    tags:
      - "v*.*"
  pull_request:
    branches: [master, develop]

jobs:

  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - uses: actions/setup-python@v4
        with:
          python-version: "3.10"

      - run: python -m pip install --upgrade pip
      - run: pip install -r requirements.txt
      - run: pip install joblib

      - name: Run fast tests (PR + develop)
        if: github.ref_name != 'master'
        run: pytest -m "not long" -v

      - name: Run full test suite (master)
        if: github.ref_name == 'master'
        run: pytest -v

  build:
    needs: test
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3
      - run: docker build -t credit-default-api .

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')

    steps:
      - uses: actions/checkout@v3

      - name: Log in to Docker Hub
        run: echo "${{ secrets.DOCKER_PASSWORD }}" | docker login -u "${{ secrets.DOCKER_USERNAME }}" --password-stdin

      - name: Build Docker image
        run: docker build -t ${{ secrets.DOCKER_USERNAME }}/credit-default-api:latest .

      - name: Push Docker image
        run: docker push ${{ secrets.DOCKER_USERNAME }}/credit-default-api:latest
        
🐙 6️⃣ Lier le repository GitHub
git remote add origin https://github.com/fremontben-prog/chap8devops.git
🧪 Lancer les tests localement
pytest -v

Tests rapides uniquement :
pytest -m "not long"

🎯 Objectif du projet
Mettre en place une architecture MLOps complète, incluant :
- Tracking d expériences
- Conteneurisation
- Monitoring
- Tests automatisés
- CI/CD
- Déploiement Docker Hub
