import torch
import pandas as pd
import re
from PIL import Image, ExifTags
from pathlib import Path
from datetime import datetime
import os
from transformers import pipeline
import src.config as cfg 

class ImageNutritionLoader:
    def __init__(self, participant_id: str, images_dir: Path):
       
        self.p_id = participant_id
        path_str = str(images_dir)
        path_hyphen = Path(path_str.replace("food_images", "food-images"))
        
        if images_dir.exists():
            self.images_dir = images_dir
        elif path_hyphen.exists():
            self.images_dir = path_hyphen
        else:
            self.images_dir = images_dir
            
        self.date_tags = [36867, 306]
        self.csv_path = cfg.DATA_DIR / "nutrition_db.csv"
        self.food_db = self._load_nutrition_db()

        local_model_path = "/app/my_local_model"

        if os.path.exists(local_model_path):
            try:
                self.classifier = pipeline("image-classification", model=local_model_path)
                self.use_ai = True
            except Exception as e:
                raise e
        else:
            raise FileNotFoundError(f"Dossier introuvable : {local_model_path}")

    def _load_nutrition_db(self) -> pd.DataFrame:
        """
        Charge la base de données nutritionnelle CSV et normalise les noms des aliments 
        pour faciliter la recherche.
        """
        if not self.csv_path.exists():
            return pd.DataFrame(columns=['name', 'calories', 'protein', 'carbs', 'fat'])
        df = pd.read_csv(self.csv_path)
        df['name'] = df['name'].str.lower().str.strip().str.replace(" ", "_")
        return df

    def _get_image_date(self, image_path: Path) -> datetime:
        """
        Extrait la date de prise de vue des métadonnées EXIF de l'image. 
        Utilise la date de modification du fichier en cas d'absence de métadonnées.
        """
        try:
            image = Image.open(image_path)
            exif_data = image._getexif()
            if exif_data:
                for tag in self.date_tags:
                    if tag in exif_data:
                        date_str = str(exif_data[tag]).strip('\x00')
                        return datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
        except: pass
        return datetime.fromtimestamp(os.path.getmtime(image_path))

    def _get_macros_from_db(self, food_name: str, estimated_weight_g: float = 300.0) -> dict:
        """
        Recherche un aliment dans la base de données et calcule les macronutriments 
        proportionnellement au poids estimé.
        """
        clean_name = food_name.lower().replace(" ", "_")
        row = self.food_db[self.food_db['name'] == clean_name]
        
        if row.empty:
            for db_name in self.food_db['name']:
                if db_name in clean_name or clean_name in db_name:
                    row = self.food_db[self.food_db['name'] == db_name]
                    break
        
        if row.empty:
            row = self.food_db[self.food_db['name'] == 'default']

        if not row.empty:
            vals = row.iloc[0]
            mult = estimated_weight_g / 100.0
            return {
                "calories": round(vals['calories'] * mult),
                "protein_g": round(vals['protein'] * mult),
                "carbs_g": round(vals['carbs'] * mult),
                "fat_g": round(vals['fat'] * mult)
            }
        return {"calories": 0, "protein_g": 0, "carbs_g": 0, "fat_g": 0}

    def process_images(self) -> pd.DataFrame:
        """
        Parcourt le dossier d'images, identifie les plats via l'IA, récupère les dates 
        et compile les données nutritionnelles dans un DataFrame.
        """
        if not self.images_dir.exists():
            return pd.DataFrame()

        records = []
        valid_extensions = {".jpg", ".jpeg", ".png", ".heic"}

        for img_file in self.images_dir.iterdir():
            if img_file.suffix.lower() in valid_extensions:
                
                date_time = self._get_image_date(img_file)
                if date_time is None: continue

                try:
                    predictions = self.classifier(str(img_file))
                    top_pred = predictions[0]
                    food_label = top_pred['label']
                    confidence = top_pred['score']

                    macros = self._get_macros_from_db(food_label)
                    normalized_date = pd.Timestamp(date_time).normalize()

                    records.append({
                        "participant_id": self.p_id,
                        "image_name": img_file.name,
                        "datetime": date_time,
                        "date": normalized_date,
                        "meal_name": food_label,
                        "confidence": round(confidence, 2),
                        **macros
                    })
                except Exception:
                    pass

        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values("datetime").reset_index(drop=True)
            
        return df