import pandas as pd
import numpy as np
import time
import src.config as cfg
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.metrics import mean_squared_error

def encode_categorical_features(df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Encode les variables catégorielles (Sexe, ID) en format numérique
    et assure le bon typage des dates pour la modélisation.
    """
    if df is None:
        input_path = cfg.DATA_DIR / "enriched_athlete_data.csv"
        if input_path.exists():
            df = pd.read_csv(input_path)
        else:
            raise FileNotFoundError(f"Fichier introuvable : {input_path}")

    df['date'] = pd.to_datetime(df['date'])

    if 'Gender' in df.columns and df['Gender'].dtype == 'object':
        df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    
    df['participant_id_code'] = df['participant_id'].astype('category').cat.codes

    output_path = cfg.DATA_DIR / "step1_types_fixed.csv"
    df.to_csv(output_path, index=False)

    return df


def remove_features(df: pd.DataFrame = None, features_to_remove: list= None) -> pd.DataFrame:
    """
    Supprime les caractéristiques spécifiées du DataFrame.
    """
    if df is None:
        input_path = cfg.DATA_DIR / "step1_types_fixed.csv"
        if input_path.exists():
            df = pd.read_csv(input_path)
        else:
            raise FileNotFoundError(f"Fichier introuvable : {input_path}")
            
    if features_to_remove is None:
        features_to_remove = ['Gender', 'bpm_error', 'stride_walk_cm', 'stride_run_cm', 'total_distance_cm','sleep_duration_h','confidence']

    df = df.drop(columns=features_to_remove, errors='ignore')
    
    output_file = cfg.DATA_DIR / "step2_cols_removed.csv"
    df.to_csv(output_file, index=False)
    return df




def clean_and_impute_simple(df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Pipeline de nettoyage rapide et robuste :
    Clipping métier, Winsorization et Imputation simple (Continuité/Médiane).
    """
    ID_COLS = ['participant_id', 'date']
    TARGET_COL = 'is_injured'

    ORDINAL_COLS = ['fatigue', 'sleep_quality', 'soreness', 'stress', 'mood']

    BOUNDED_COLS = {
        'sleep_duration_h': (0, 24),
        'sleep_efficiency': (0, 100),
        'readiness': (0, 10),
        'confidence': (0, 100),
        'resting_bpm': (40, 120)
    }

    WINSOR_COLS = [
        'training_load', 'training_duration_min',
        'total_steps', 'total_distance_cm',
        'sport_duration_min', 'sport_calories',
        'stride_walk_cm', 'stride_run_cm',
    ]

    if df is None:
        input_file = cfg.DATA_DIR / "step2_cols_removed.csv"
        if input_file.exists():
            df = pd.read_csv(input_file)
        else:
            raise FileNotFoundError(f"Fichier introuvable : {input_file}")

    df['date'] = pd.to_datetime(df['date'])
    df_clean = df.copy()

    for col, (low, high) in BOUNDED_COLS.items():
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].clip(lower=low, upper=high)

    for pid in df_clean['participant_id'].unique():
        mask = df_clean['participant_id'] == pid
        sub = df_clean.loc[mask]

        if len(sub) < 10:
            continue

        for col in WINSOR_COLS:
            if col in df_clean.columns and sub[col].notna().sum() > 5:
                q_low = sub[col].quantile(0.01)
                q_high = sub[col].quantile(0.99)
                df_clean.loc[mask, col] = sub[col].clip(q_low, q_high)

    df_clean = df_clean.sort_values(['participant_id', 'date'])

    cols_ord = [c for c in ORDINAL_COLS if c in df_clean.columns]
    if cols_ord:
        df_clean[cols_ord] = (
            df_clean
            .groupby('participant_id')[cols_ord]
            .transform(lambda x: x.ffill().bfill())
        )

    numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns
    cols_to_fill = [c for c in numeric_cols if c not in ID_COLS + [TARGET_COL]]

    df_clean[cols_to_fill] = (
        df_clean
        .groupby('participant_id')[cols_to_fill]
        .transform(lambda x: x.fillna(x.median()))
    )

    remaining_nans = df_clean[cols_to_fill].isnull().sum().sum()
    if remaining_nans > 0:
        df_clean[cols_to_fill] = df_clean[cols_to_fill].fillna(df_clean[cols_to_fill].median())

    output_path = cfg.DATA_DIR / "test_dataset.csv"
    df_clean.to_csv(output_path, index=False)
    
    return df_clean


def clean_outliers_and_impute_Mice_KNN(df: pd.DataFrame = None, method: str = 'MICE') -> pd.DataFrame:
    """
    Pipeline de nettoyage avancé et d'imputation par participant.
    Gère les contraintes physiques, les données ordinales et les valeurs extrêmes.
    """
    ID_COLS = ['participant_id', 'participant_id_code', 'date']
    TARGET_COL = 'is_injured'

    ORDINAL_COLS = ['fatigue', 'sleep_quality', 'soreness', 'stress', 'mood']

    BOUNDED_COLS = {
        'sleep_duration_h': (0, 24),
        'sleep_efficiency': (0, 100),
        'readiness': (0, 10),
        'confidence': (0, 100),
        'resting_bpm': (40, 120)
    }

    WINSOR_COLS = [
        'training_load', 'training_duration_min',
        'total_steps', 'total_distance_cm',
        'sport_duration_min', 'sport_calories',
        'stride_walk_cm', 'stride_run_cm',
        'resting_bpm'
    ]

    if df is None:
        input_path = cfg.DATA_DIR / "step2_cols_removed.csv"
        if input_path.exists():
            df = pd.read_csv(input_path)
        else:
            raise FileNotFoundError(f"Fichier introuvable : {input_path}")

    df['date'] = pd.to_datetime(df['date'])
    participants = df['participant_id'].unique()

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    cols_to_impute = [c for c in numeric_cols if c not in ID_COLS and c != TARGET_COL]

    for col, (min_v, max_v) in BOUNDED_COLS.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=min_v, upper=max_v)

    df_capped_list = []
    for p in participants:
        sub_df = df[df['participant_id'] == p].copy()
        
        if len(sub_df) > 10:
            for col in WINSOR_COLS:
                if col in sub_df.columns and sub_df[col].notna().sum() > 5:
                    lower = sub_df[col].quantile(0.01)
                    upper = sub_df[col].quantile(0.99)
                    sub_df[col] = sub_df[col].clip(lower=lower, upper=upper)
        
        df_capped_list.append(sub_df)

    df_capped = pd.concat(df_capped_list)

    if method == 'KNN':
        imputer_algo = KNNImputer(n_neighbors=5)
    else:
        imputer_algo = IterativeImputer(max_iter=100, random_state=42, min_value=0)

    df_final_list = []

    for p in participants:
        sub_df = df_capped[df_capped['participant_id'] == p].copy()
        sub_df = sub_df.sort_values('date') 
        
        sub_numeric = sub_df[cols_to_impute]
        
        if sub_numeric.isnull().sum().sum() > 0:
            imputed_data = imputer_algo.fit_transform(sub_numeric)
            sub_df[cols_to_impute] = pd.DataFrame(imputed_data, columns=cols_to_impute, index=sub_numeric.index)
        
        df_final_list.append(sub_df)

    df_final = pd.concat(df_final_list)

    for col in ORDINAL_COLS:
        if col in df_final.columns:
            df_final[col] = df_final[col].round().astype('Int64')

    for col, (min_v, max_v) in BOUNDED_COLS.items():
        if col in df_final.columns:
            df_final[col] = df_final[col].clip(lower=min_v, upper=max_v)

    df_final = df_final.fillna(0)

    output_path = cfg.DATA_DIR / "test_dataset.csv"
    df_final.to_csv(output_path, index=False)

    return df_final

def verify_data_quality(df: pd.DataFrame = None) -> bool:
    """
    Audit final du dataset :
    Vérifie les valeurs manquantes et le respect des règles physiques.
    """
    if df is None:
        file_path = cfg.DATA_DIR / "test_dataset.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
        else:
            raise FileNotFoundError(f"Fichier introuvable : {file_path}")

    rules = {
        'calories': [500, 15000, True],
        'training_load': [0, 3000, True],
        'total_steps': [0, 100000, True],
        'resting_bpm': [30, 120, True],
        'sleep_efficiency': [0, 100, True],
        'sleep_duration_h': [0, 24, True],
        'energy_balance': [-5000, 5000, False],
        'acwr': [0, 5, True],
        'fatigue': [0, 10, True],
        'stress': [0, 100, True],
        'resting_bpm': [40, 120, True],
    }

    global_error = False
    
    total_nans = df.isnull().sum().sum()
    if total_nans != 0:
        global_error = True

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

    for col in numeric_cols:
        n_missing = df[col].isnull().sum()
        
        if not df[col].empty:
            val_min = df[col].min()
            val_max = df[col].max()
        else:
            val_min, val_max = 0, 0
        
        status = "OK"
        
        if n_missing > 0:
            status = "NAN TROUVE"
            global_error = True
        
        if col in rules:
            limit_min, limit_max, must_be_positive = rules[col]
            
            if must_be_positive and val_min < 0:
                status = f"NEGATIF ({val_min:.2f})"
                global_error = True
            elif val_max > limit_max * 1.5: 
                status = f"TROP HAUT ({val_max:.2f})"
            elif val_min < limit_min and limit_min > 0:
                status = f"TROP BAS ({val_min:.2f})"

        print(f"{col:<25} | {n_missing:<8} | {val_min:<10.2f} | {val_max:<10.2f} | {status}")

    if global_error:
        return False
    else:
        return True


def run_real_world_benchmark(df_source: pd.DataFrame = None):
    """
    Compare les 3 fonctions de gestion de valeurs manquantes.
    """
    if df_source is None:
        input_file = cfg.DATA_DIR / "step2_cols_removed.csv"
        if input_file.exists():
            df_source = pd.read_csv(input_file)
        else:
            raise FileNotFoundError(f"Fichier introuvable : {input_file}")

    numeric_cols = df_source.select_dtypes(include=['float64', 'int64']).columns
    exclude = ['participant_id', 'participant_id_code', 'date', 'is_injured', 'Gender']
    candidates = [c for c in numeric_cols if c not in exclude]

    good_cols = []
    for col in candidates:
        fill_rate = df_source[col].notna().mean()
        if fill_rate > 0.5:
            good_cols.append(col)
            
    df_truth = df_source.dropna(subset=good_cols).copy()
    
    if len(df_truth) < 50:
        top_3_cols = df_source[candidates].count().sort_values(ascending=False).head(3).index.tolist()
        df_truth = df_source.dropna(subset=top_3_cols).copy()
        good_cols = top_3_cols
        
        if len(df_truth) < 10:
             return "MANUEL"

    np.random.seed(42)
    df_masked = df_truth.copy()
    
    mask = np.random.choice([True, False], size=df_truth[good_cols].shape, p=[0.2, 0.8])
    
    vals = df_masked[good_cols].values
    vals[mask] = np.nan
    df_masked[good_cols] = vals

    strategies = {
        "1. SIMPLE (Fill/Median)": lambda df: clean_and_impute_simple(df),
        "2. KNN (Voisins)       ": lambda df: clean_outliers_and_impute_Mice_KNN(df, method='KNN'),
        "3. MICE (Iterative)    ": lambda df: clean_outliers_and_impute_Mice_KNN(df, method='MICE')
    }

    results = {}

    print("-" * 65)
    print(f"{'MÉTHODE':<25} | {'RMSE (Erreur)':<15} | {'TEMPS'}")
    print("-" * 65)

    for name, func in strategies.items():
        df_input = df_masked.copy() 
        
        try:
            start_time = time.time()
            df_repaired = func(df_input)
            
            elapsed = time.time() - start_time
            truth_vals = df_truth[good_cols].values[mask]
            
            if not all(col in df_repaired.columns for col in good_cols):
                 raise ValueError("Colonnes perdues pendant le nettoyage")

            repaired_vals = df_repaired[good_cols].values[mask]
            
            rmse = np.sqrt(mean_squared_error(truth_vals, repaired_vals))
            results[name] = rmse
            
            print(f"{name:<25} | {rmse:.4f}          | {elapsed:.2f} s")
            
        except Exception as e:
            print(f"{name:<25} | ERREUR ({str(e)})")
            results[name] = float('inf')
    print("-" * 65)
    best_method = min(results, key=results.get)
    print(f"\n MEILLEURE MÉTHODE : {best_method}")
    return best_method



def apply_final_domain_corrections(df: pd.DataFrame = None) -> pd.DataFrame:
    """
    DERNIÈRE ÉTAPE : Correction des valeurs physiologiquement impossibles.
    """
    if df is None:
        input_path = cfg.DATA_DIR / "FINAL_READY_FOR_ML_ADVANCED.csv"
        if input_path.exists():
            df = pd.read_csv(input_path)
        else:
            raise FileNotFoundError(f"Fichier introuvable : {input_path}")

    # Correction BPM (Cœur arrêté = 0 -> Médiane)
    df['resting_bpm'] = (
    df['resting_bpm']
    .replace(0, np.nan)
    .fillna(df.groupby('participant_id')['resting_bpm'].transform('median'))
    )

    df['calories'] = df['calories'].clip(lower=500, upper=5000)
    df['energy_balance'] = df['energy_balance'].clip(lower=-2000, upper=2000)

    output_path = cfg.DATA_DIR / "test_dataset.csv"
    df.to_csv(output_path, index=False)
    return df














