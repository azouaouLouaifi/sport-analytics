# Analyse de Performance & Prediction de Blessure

Ce projet propose une solution complete pour l'analyse de donnees de lifelogging (Fitbit), de questionnaires de bien-etre (PMSys) et de nutrition (Images). L'objectif est de predire le risque de blessure et d'etablir un indice de performance personnalise pour les athletes.

Le projet est entierement conteneurise via Docker pour garantir une execution identique et sans friction sur Windows, macOS et Linux.

## Architecture du Projet

.
├── data/                  # Dossier des donnees
│   ├── p01/               # Donnees du participant 01 (A TELECHARGER)
│   ├── p03/               # ...
│   ├── p05/               # ...
│   ├── test_dataset.csv   # Donnee pour la modelisation (Incluse)
│   └── participant-overview.xlsx
├── my_local_model/        # Modele Hugging Face (Local)
├── notebooks/             # Jupyter Notebooks
├── src/                   # Code source Python
│   ├── config.py          # Configuration globale
│   ├── data_loader.py     # Ingestion des donnees
│   ├── nutrition_loader.py# Traitement des images
│   ├── preparation_data.py# Nettoyage & Feature Engineering
│   └── modelisation.py    # Modeles (Performance & Blessure)
├── Dockerfile             # Configuration Docker
├── requirements.txt       # Dependances Python
└── README.md              # Documentation du projet

## Prerequis

* Docker Desktop (ou Docker Engine sur Linux)
* Git

## Installation et Demarrage

### 1. Cloner le depot

Recuperez le code source sur votre machine :

git clone <URL_DE_VOTRE_REPO_GIT>
cd <NOM_DU_DOSSIER>

### 2. Ajout des donnees (Etape Critique)

Les donnees brutes (images, jsons) sont trop volumineuses pour Github.

1. Telechargez l'archive des donnees ici : https://we.tl/t-EZFbVJCUj1
2. Decompressez le fichier.
3. Placez le contenu (dossiers p01, p03, p05 et fichier excel) dans le dossier `data/` du projet.

Votre dossier `data/` doit contenir a la fin :
- p01/
- p03/
- p05/
- participant-overview.xlsx
- test_dataset.csv
### 3. Installation du Modèle (Hugging Face)

Pour que l'analyse d'images fonctionne hors-ligne, le fichier binaire du modèle est requis.

1. Téléchargez le fichier `pytorch_model.bin` depuis ce lien : https://huggingface.co/nateraw/food/tree/main
2. Placez ce fichier dans le dossier `my_local_model/` situé à la racine du projet.

### 4. Lancer l'environnement

Ouvrez votre terminal a la racine du projet.

Windows / macOS / Linux :

docker compose up --build

## Acceder au Projet

Une fois le serveur lance :
1. Ouvrez votre navigateur web.
2. Accedez a l'URL : http://localhost:8888

* Pour lancer l'analyse : Ouvrez le notebook notebooks/code.ipynb.

## Approche Scientifique & Technique

### 1. Ingestion et Nettoyage (ETL)
* Pipeline Oriente Objet pour standardiser JSON, CSV, Excel.
* Alignement temporel strict des donnees physiologiques et declaratives.
* Computer Vision : Estimation calorique via Deep Learning (Hugging Face) sur images de repas.

### 2. Qualite des Donnees
* Winsorization : Traitement statistique des valeurs aberrantes sans suppression destructive.
* Imputation MICE : Comblement intelligent des donnees manquantes.

### 3. Modelisation Predictive (XGBoost)
* Indice de Performance (IDP) : Variable cible synthetique (Banister + Hooper + Halson).
* Prediction de Blessure : Classification optimisee "Small Data".
    * Strategie : Fenetre temporelle (3 jours) + Cost-Sensitive Learning.
    * Resultats : Recall de 80% (Securite) avec peu de fausses alertes.

## Commandes Utiles

* Arreter le serveur : Ctrl + C
* Nettoyer les conteneurs : docker compose down