import json
import ast
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Union
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report, confusion_matrix, precision_recall_curve
import src.config as cfg

def robust_z_score(x):
    if x.std(ddof=0) == 0:
        return np.zeros_like(x)
    return (x - x.mean()) / x.std(ddof=0)

def minmax_scale_custom(x):
    if x.max() == x.min():
        return pd.Series([50]*len(x), index=x.index)
    return (x - x.min()) / (x.max() - x.min()) * 100

def create_performance_index(df: pd.DataFrame = None) -> pd.DataFrame:
    if df is None:
        file_path = cfg.DATA_DIR / "test_dataset.csv"
        df = pd.read_csv(file_path)
    
    df_clean = df.copy()

    df_clean['hooper_negative_sum'] = df_clean['fatigue'] + df_clean['stress'] + df_clean['soreness']
    

    grouper = df_clean.groupby('participant_id')

    df_clean['z_chronic'] = grouper['chronic_load'].transform(robust_z_score)
    df_clean['z_sleep_eff'] = grouper['sleep_efficiency'].transform(robust_z_score)
    df_clean['z_bpm_inv'] = grouper['resting_bpm'].transform(robust_z_score) * -1
    df_clean['z_hooper'] = grouper['hooper_negative_sum'].transform(robust_z_score)
    df_clean['z_acute'] = grouper['acute_load'].transform(robust_z_score)

    w_fit = 0.35
    w_rec = 0.35
    w_fat = 0.30

    score_fitness = df_clean['z_chronic']
    score_recovery = (df_clean['z_sleep_eff'] + df_clean['z_bpm_inv']) / 2
    score_fatigue = (df_clean['z_hooper'] + df_clean['z_acute']) / 2

    df_clean['IDP_raw'] = (w_fit * score_fitness) + (w_rec * score_recovery) - (w_fat * score_fatigue)

    df_clean['IDP'] = df_clean.groupby('participant_id')['IDP_raw'].transform(minmax_scale_custom)

    if 'is_injured' in df_clean.columns:
        df_clean.loc[df_clean['is_injured'] == 1, 'IDP'] = 0

    cols_to_drop = [
        'hooper_negative_sum', 'z_chronic', 'z_sleep_eff', 
        'z_bpm_inv', 'z_hooper', 'z_acute', 'IDP_raw'
    ]
    df_final = df_clean.drop(columns=cols_to_drop)

    output_path = cfg.DATA_DIR / "FINAL_DATASET_WITH_IDP.csv"
    df_final.to_csv(output_path, index=False)
    
    print("Termine. IDP calcule, Blessures gerees")
    print(f"Dimensions finales : {df_final.shape}")
    print(f"Apercu IDP moyen : \n{df_final.groupby('participant_id')['IDP'].mean()}")

    return df_final

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.sort_values(['participant_id', 'date'])
    
    features_to_lag = [
        'IDP',
        'sleep_efficiency',
        'resting_bpm',
        'training_load',
        'acwr',
        'fatigue',
        'stress',
        'soreness',
        'mood',
        'energy_balance',
        'readiness',
        'alcohol_consumed',
        'sleep_duration_min'        
    ]
    
    features_to_lag = [f for f in features_to_lag if f in df.columns]

    for col in features_to_lag:
        df[f'{col}_lag1'] = df.groupby('participant_id')[col].shift(1)
        df[f'{col}_lag2'] = df.groupby('participant_id')[col].shift(2)
        df[f'{col}_rolling7'] = df.groupby('participant_id')[col].transform(lambda x: x.rolling(7).mean().shift(1))

    df_final = df.dropna().reset_index(drop=True)
    
    output_path = cfg.DATA_DIR / "FINAL_DATASET_WITH_IDP_AND_LAGS.csv"
    df_final.to_csv(output_path, index=False)
    
    print(f"Lags generes. Dataset final sauvegarde : {output_path}")
    print(f"Dimensions finales : {df_final.shape}")
    
    return df_final

def prepare_X_y_for_training(df: pd.DataFrame, target_col: str):
    y = df[target_col]
    
    keywords_ok = ['_lag', '_rolling', 'Age', 'Height', 'weight', 'participant_id_code', 'benchmark_5km_speed_kmh']
    
    features_cols = [c for c in df.columns if any(k in c for k in keywords_ok)]
    
    if target_col in features_cols:
        features_cols.remove(target_col)
        
    X = df[features_cols]
    
    print(f"Cible : {target_col}")
    print(f"Features selectionnees ({len(features_cols)}) :")
    print(features_cols)
    print("Variables supprimees  :")
    deleted = [c for c in df.columns if c not in features_cols and c != target_col]
    print(deleted)
    
    return X, y

def train_performance_model(df: pd.DataFrame):

    X, y = prepare_X_y_for_training(df, target_col='IDP')

    print(f"Dimensions : {X.shape}")

    split_index = int(len(X) * 0.80)
    
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    print(f"Training set : {X_train.shape[0]} lignes")
    print(f"Test set     : {X_test.shape[0]} lignes (Futur inconnu)")

    tscv = TimeSeriesSplit(n_splits=5)

    xgb_reg = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=10
    )

    param_grid = {
        'n_estimators': [100, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05],
        'subsample': [0.8]
    }

    grid_search = GridSearchCV(
        estimator=xgb_reg,
        param_grid=param_grid,
        scoring='neg_mean_absolute_error',
        cv=tscv,
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    best_model = grid_search.best_estimator_
    print(f"Meilleurs parametres : {grid_search.best_params_}")

    y_pred = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("-" * 40)
    print("RESULTATS SUR DONNEES INCONNUES (TEST SET)")
    print("-" * 40)
    print(f"MAE  (Erreur Moyenne) : {mae:.2f} / 100")
    print(f"RMSE (Erreur Pic)     : {rmse:.2f}")
    print(f"R2   (Precision)      : {r2:.2f}")
    print("-" * 40)

    plt.figure(figsize=(10, 6))
    xgb.plot_importance(best_model, max_num_features=12, height=0.5, title="Facteurs Cles de la Performance ")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 6))
    plt.plot(y_test.values[:100], label='Vraie Perf (IDP)', color='black', alpha=0.7)
    plt.plot(y_pred[:100], label='Prediction IA', color='#e74c3c', linestyle='--')
    plt.title("Capacite du modele a suivre la forme des joueurs (Echantillon Test)")
    plt.ylabel("Indice IDP (0-100)")
    plt.xlabel("Jours (Test Set)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return best_model

def create_future_target(df, window=3):
    df = df.sort_values(['participant_id', 'date'])
    
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=window)
    
    target_col = f'is_injured_next_{window}d'
    df[target_col] = df.groupby('participant_id')['is_injured']\
                       .rolling(window=indexer).max()\
                       .reset_index(0, drop=True)
    
    df = df.dropna(subset=[target_col])
    
    return df, target_col

def train_injury_model(df: pd.DataFrame):

    df_window, target_col = create_future_target(df, window=3)
    X, y = prepare_X_y_for_training(df_window, target_col=target_col)

    n_injuries = y.sum()
    print(f"Stats Cible ({target_col}) : {n_injuries} jours a risque sur {len(y)} echantillons")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.20,
        stratify=y,
        shuffle=True,
        random_state=42
    )
    
    print(f"Train Set : {y_train.sum()} blessures")
    print(f"Test Set  : {y_test.sum()} blessures")

    pos = y_train.sum()
    neg = len(y_train) - pos
    spw = neg / max(pos, 1)
    
    xgb_clf = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=spw,
        eval_metric='logloss',
        n_jobs=-1,
        random_state=42,
    )

    param_grid = {
        'max_depth': [3, 4],
        'learning_rate': [0.01, 0.05],
        'n_estimators': [100, 200],
        'min_child_weight': [2],
        'subsample': [0.8]
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=xgb_clf,
        param_grid=param_grid,
        scoring='f1',
        cv=cv,
        verbose=0,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    y_proba = best_model.predict_proba(X_test)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    
    target_recall = 0.80
    
    valid_indices = np.where(recall >= target_recall)[0]
    
    if len(valid_indices) > 0:
        best_idx = valid_indices[np.argmax(precision[valid_indices])]
        best_thresh = thresholds[best_idx]
        print(f"Strategie : Recall Cible >= {target_recall*100}%")
    else:
        print("Strategie : Max F1 (Fallback)")
        fscore = (2 * precision * recall) / (precision + recall + 1e-6)
        best_idx = np.argmax(fscore)
        best_thresh = thresholds[best_idx]
    
    print(f"Seuil Optimal choisi : {best_thresh:.4f}")

    y_pred_opt = (y_proba >= best_thresh).astype(int)

    cm = confusion_matrix(y_test, y_pred_opt)
    tn, fp, fn, tp = cm.ravel()

    print("RESULTAT FINAL :")
    print(f"Sains bien detectes      : {tn}")
    print(f"Fausses Alertes (Erreur) : {fp}")
    print(f"Blessures RATEES (Erreur): {fn}")
    print(f"Blessures DETECTEES      : {tp}")

    print("\n", classification_report(y_test, y_pred_opt))
    
    return best_model