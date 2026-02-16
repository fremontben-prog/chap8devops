
mlops-project/
│
├── api/
│   ├── main.py
│   ├── model.pkl
│   ├── requirements.txt
│   └── Dockerfile
│
├── monitoring/
│   ├── monitoring.py
│   └── drift_report.html
│
├── data/
│   └── train.csv
│
├── .github/workflows/
│   └── ci.yml
│
├── docker-compose.yml 
│
└── README.md


# Pour l'API
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000


Elasticsearch → http://localhost:9200

Kibana → http://localhost:5601


données clés à logger
📌 Données modèle

input_features

prediction

probability

model_version

📌 Données opérationnelles

timestamp

execution_time_ms

status_code

error_message