# Image de base légère
FROM python:3.9-slim

# Optimisations Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Installation des dépendances système
# CORRECTION ICI : On a remplacé libgl1-mesa-glx par libgl1
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Dossier de travail dans le conteneur
WORKDIR /app

# Copie et installation des librairies Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Commande par défaut (Jupyter Lab)
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]