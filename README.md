# Détecteur de Race de Chien (Deep Learning CNN)

Ce projet comprend une partie Recherche & Entraînement (Jupyter Notebooks) comparant plusieurs architectures (MobileNetV2, EfficientNetB0, CNN "Maison") et une partie Application (API FastAPI + Interface Web).

# Structure du Projet
```Plaintext

.
├── Projet-Application-CNN-DeepLearning   # Partie Déploiement (API & Web)
│   ├── api
│   │   ├── main.py                       # Serveur Backend (FastAPI)
│   │   ├── class_indices.json            # Mapping ID <-> Nom de race
│   │   ├── models/                       # Dossier où déposer vos modèles .keras
│   │   └── requirements.txt              # Dépendances de l'API
│   └── application
│       └── index.html                    # Interface Utilisateur (Frontend)
│
└── Projet-Etude-CNN-DeepLearning         # Partie Data Science (Entraînement)
    ├── dataset                           # Dossier (vide) pour les données Kaggle
    └── notebooks
        ├── notebook_dog_breed_mobileNetv2.ipynb
        ├── notebook_dog_breed_EfficientNetB0.ipynb
        ├── notebook_dog_breed_fromScratch.ipynb
        └── models/                       # Où les modèles entraînés sont sauvegardés
```

# Prérequis

- Python 3.10+ (Recommandé pour la compatibilité GPU/TensorFlow).
- Anaconda / Miniconda (Conseillé pour la gestion des environnements).
- Carte Graphique NVIDIA (Fortement recommandée pour l'entraînement).

# Installation des dépendances
-  Clonez ce dépôt.
- Installez les librairies nécessaires :
```Bash

pip install -r Projet-Application-CNN-DeepLearning/api/requirements.txt
# Et pour la partie notebooks :
pip install tensorflow pandas matplotlib jupyter
``` 
# 1. Récupération du Dataset

Le jeu de données étant trop volumineux pour GitHub, il doit être téléchargé manuellement depuis Kaggle.
- Rendez-vous sur : Kaggle - Dog Breed Identification.
- Téléchargez le fichier dog-breed-identification.zip.
- Extrayez le contenu dans le dossier Projet-Etude-CNN-DeepLearning/dataset/.

Votre structure de fichiers doit ressembler exactement à ceci pour que les notebooks fonctionnent :
```Plaintext

Projet-Etude-CNN-DeepLearning/
└── dataset/
    └── Dog-Breed-Identification/
        ├── labels.csv
        ├── train/         # Contient les images 000a...jpg
        └── test/          # (Optionnel)
```
# 2. Entraînement des Modèles (Partie Étude)

Cette phase permet de créer les cerveaux de l'IA. Ouvrez les notebooks situés dans `Projet-Etude-CNN-DeepLearning/notebooks/` avec Jupyter ou VS Code.

Trois approches sont disponibles :
- notebook_dog_breed_mobileNetv2.ipynb : Transfer Learning avec MobileNetV2 (Rapide et léger, idéal pour débuter).
- notebook_dog_breed_EfficientNetB0.ipynb : Transfer Learning avec EfficientNetB0 (Meilleur compromis précision/vitesse).
- notebook_dog_breed_fromScratch.ipynb : Création d'un réseau CNN couche par couche (Pour comprendre la théorie, moins performant).

**Résultat :** À la fin de l'exécution, les modèles .keras (ex: model_dog_efficientnetb0.keras) et le fichier class_indices.json seront générés dans le dossier `notebooks/` ou `notebooks/models/`.

# 3. Lancement de l'Application

Une fois les modèles entraînés, vous pouvez lancer l'interface web.
## Étape A : Préparation des fichiers

Pour que l'API fonctionne, vous devez déplacer les fichiers générés vers le dossier de l'application :
- Copiez vos modèles .keras depuis la partie Etude vers Projet-Application-CNN-DeepLearning/api/models/.
- Assurez-vous que class_indices.json est bien présent dans Projet-Application-CNN-DeepLearning/api/.

## Étape B : Démarrer l'API (Backend)

Ouvrez un terminal dans le dossier Projet-Application-CNN-DeepLearning/api/ et lancez :
```Bash
uvicorn main:app --reload
```

Le terminal doit afficher : `Uvicorn running on http://127.0.0.1:8000`

## Étape C : Utiliser l'Interface (Frontend)
- Allez dans le dossier Projet-Application-CNN-DeepLearning/application/.
- Ouvrez simplement le fichier index.html dans votre navigateur (double-clic).


### Fonctionnalités de l'App :
- Menu déroulant : Choisissez quel modèle utiliser (MobileNet, EfficientNet, Scratch...) parmi ceux présents dans le dossier models/.
- Upload : Envoyez une image depuis votre ordinateur.
- URL : Collez le lien d'une image internet.
- Résultats : Affiche le Top 3 des races prédites avec les barres de confiance (Vert/Orange/Rouge).

# Dépannage courant
- Erreur CORS : Si l'interface web ne réagit pas, vérifiez que le serveur uvicorn tourne bien.
- Erreur OOM (Out Of Memory) : Si l'entraînement plante, réduisez le `BATCH_SIZE` (passez de 32 à 16 ou 8) dans les notebooks.
- Modèle introuvable : Vérifiez que le nom du fichier `.keras` dans le dossier `api/models/` correspond exactement à ce qui s'affiche dans le menu déroulant.