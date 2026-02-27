# monitoring/middleware.py
from fastapi import Request
from datetime import datetime
import time
from elasticsearch import Elasticsearch
import os
import json
import psutil

# Config Elasticsearch
ES_HOST = os.getenv("ES_HOST", "localhost")  # nom du service Docker
es = Elasticsearch(f"http://{ES_HOST}:9200")

INDEX_PROD = os.getenv("INDEX_PROD", "predictions")

def setup_middleware(app):
    # Exclusion de run-drift car traitement différent
    EXCLUDED_PATHS = ["/run-drift"]
    
    @app.middleware("http")
    async def log_all_requests(request: Request, call_next):
        
        # Si route exclue, rien n'est loggé
        if request.url.path in EXCLUDED_PATHS:
            return await call_next(request)
        
        
        start_time = time.time()
                
        body_data = None
        status_code = None

        # Lecture body si JSON
        try:
            body_bytes = await request.body()
            body_data = json.loads(body_bytes.decode("utf-8")) if body_bytes else None
        except Exception:
            body_data = None

        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            # Erreur inattendue (500)
            response = None
            status_code = 500
            body_data = {"error": str(e)}

        process_time = (time.time() - start_time) * 1000
        
        # USage ressources CPU et mémoire
        cpu_usage = psutil.cpu_percent(interval=None)
        memory_usage = psutil.virtual_memory().percent

        # Log complet
        log_data = {
            "@timestamp": datetime.now().isoformat(),
            "path": request.url.path,
            "method": request.method,
            "status_code": status_code,
            "execution_time_ms": process_time,
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "request_body": body_data
        }

        # Essaye d’envoyer dans Elasticsearch sans bloquer
        try:
            print(f"Log DATA : {log_data}")
            print(f"Index prod : {INDEX_PROD}")
            es.index(index=INDEX_PROD, document=log_data)
        except Exception as e:
            print(f"Erreur Elasticsearch middleware : {e}")


        # Retourne la réponse au client
        if response:
            return response
        else:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=status_code or 500,
                content={"detail": "Erreur interne"}
            )