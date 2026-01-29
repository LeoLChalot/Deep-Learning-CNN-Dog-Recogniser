from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import json
import requests
import os

app = FastAPI(title="Dog Breed API Multi-Model 🐶")

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURATION ---
MODELS_DIR = "models" 

# Dictionnaire pour stocker les modèles
loaded_models = {}

# Charger les labels
try:
    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)
        labels_map = {int(k): v for k, v in class_indices.items()}
    print("Labels chargés.")
except FileNotFoundError:
    print("ERREUR : fichier des labels introuvable.")
    labels_map = {}

def get_model(model_name: str):
    """
    Charge le modèle depuis 'models/' s'il n'est pas en mémoire.
    """
    # 1. Construire le chemin complet
    safe_model_name = os.path.basename(model_name)
    model_path = os.path.join(MODELS_DIR, safe_model_name)

    # 2. Vérifier si le fichier existe
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Le modèle '{safe_model_name}' est introuvable dans le dossier '{MODELS_DIR}'.")

    # 3. Charger le modèle s'il n'est pas dans "loaaded_models"
    if safe_model_name not in loaded_models:
        print(f"Chargement du modèle depuis {model_path}.")
        try:
            loaded_models[safe_model_name] = load_model(model_path)
            print(f"{safe_model_name} prêt.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erreur chargement modèle : {str(e)}")
    
    return loaded_models[safe_model_name]

# --- FONCTIONS UTILITAIRES ---
def prepare_image(img_pil):
    img_pil = img_pil.resize((224, 224))
    img_array = image.img_to_array(img_pil)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_process(model, img_array):
    predictions = model.predict(img_array)
    
    # predictions[0] contient les 120 probabilités
    probs = predictions[0]
    
    # 1. np.argsort trie les indices du plus petit score au plus grand
    # 2. [-3:] prend les 3 derniers (donc les 3 plus grands)
    # 3. [::-1] inverse la liste pour avoir le meilleur en premier
    top_3_indices = np.argsort(probs)[-3:][::-1]
    
    results = []
    
    for i in top_3_indices:
        confidence = float(probs[i])
        # On ignore les résultats vraiment trop faibles (optionnel, ex: < 0.01%)
        if confidence > 0.0001: 
            results.append({
                "breed": labels_map.get(int(i), "Inconnu"),
                "confidence": round(confidence * 100, 2)
            })
            
    # On retourne un objet contenant la liste
    return {"predictions": results}

class UrlRequest(BaseModel):
    url: str
    model_name: str

# --- ROUTES API ---
@app.post("/predict/file")
async def predict_file(
    file: UploadFile = File(...), 
    model_name: str = Query(..., description="Nom du fichier modèle .keras")
):
    """Prédire depuis un fichier avec choix du modèle"""
    target_model = get_model(model_name)
    
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    processed_img = prepare_image(img)
    
    return predict_process(target_model, processed_img)

@app.post("/predict/url")
async def predict_url(request: UrlRequest):
    """Prédire depuis une URL avec choix du modèle"""
    target_model = get_model(request.model_name)
    
    try:
        response = requests.get(request.url, timeout=10)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content)).convert('RGB')
        processed_img = prepare_image(img)
        return predict_process(target_model, processed_img)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur URL : {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)