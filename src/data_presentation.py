import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import src.config as cfg

def load_merge_and_audit(
    global_path: Path = cfg.DATA_DIR / "GLOBAL_DATASET_FULL.csv",
    nutrition_path: Path = cfg.DATA_DIR / "nutrition_daily_all_participants.csv",
    plot: bool = True
) -> pd.DataFrame:
    
    if not global_path.exists() or not nutrition_path.exists():
        raise FileNotFoundError("Fichiers source introuvables. Vérifiez les chemins dans config.py")

    df_global = pd.read_csv(global_path)
    df_nutrition = pd.read_csv(nutrition_path)

    df_global['date'] = pd.to_datetime(df_global['date'])
    df_nutrition['date'] = pd.to_datetime(df_nutrition['date'])

    df = pd.merge(df_global, df_nutrition, on=['participant_id', 'date'], how='outer')
    df = df.sort_values(by=['participant_id', 'date']).reset_index(drop=True)

    if plot:
        plt.style.use('ggplot')
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Audit Initial des Données (Avant Nettoyage)', fontsize=16)

        sns.countplot(
            data=df, x='is_injured', ax=axes[0, 0], palette='viridis'
        )
        axes[0, 0].set_title('Déséquilibre de la cible (Blessures)')
        axes[0, 0].set_xlabel('0 = Sain | 1 = Blessé')

        sns.scatterplot(
            data=df, x='training_load', y='fatigue',
            hue='is_injured', alpha=0.6, ax=axes[0, 1]
        )
        axes[0, 1].set_title("Charge d'entraînement vs Fatigue")

        sns.scatterplot(
            data=df, x='sleep_duration_h', y='fatigue',
            hue='is_injured', alpha=0.6, ax=axes[1, 0]
        )
        axes[1, 0].set_title("Durée de sommeil vs Fatigue")

        sns.regplot(
            data=df.dropna(subset=['sleep_minutes_deep', 'overall_score']),
            x='sleep_minutes_deep', y='overall_score',
            scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'}, ax=axes[1, 1]
        )
        axes[1, 1].set_title("Sommeil profond vs Score global (Oura/Garmin)")

        plt.tight_layout()
        plt.show()
    
    return df

def run_feature_engineering_pipeline(df: pd.DataFrame, plot: bool = True) -> pd.DataFrame:
    
    df = df.sort_values(['participant_id', 'date']).reset_index(drop=True)

    if 'calories' in df.columns and 'total_calories' in df.columns:
        df['energy_balance'] = df['calories'] - df['total_calories']

    df['acute_load'] = df.groupby('participant_id')['training_load'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )
    
    df['chronic_load'] = df.groupby('participant_id')['training_load'].transform(
        lambda x: x.rolling(window=28, min_periods=1).mean()
    )

    df['acwr'] = df['acute_load'] / (df['chronic_load'] + 1e-5)

    df['fatigue_3d_avg'] = df.groupby('participant_id')['fatigue'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )

    if plot:
        plt.style.use('ggplot')

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        if 'resting_bpm' in df.columns:
            sns.boxplot(x='participant_id', y='resting_bpm', data=df, ax=axes[0])
            axes[0].set_title("Variabilité du Rythme Cardiaque au Repos")
        
        if 'sleep_efficiency' in df.columns:
            sns.boxplot(x='participant_id', y='sleep_efficiency', data=df, ax=axes[1])
            axes[1].set_title("Variabilité de l'Efficacité du Sommeil")
        
        plt.tight_layout()
        plt.show()

        participants_focus = ['p01', 'p03', 'p05']
        
        for pid in participants_focus:
            if pid in df['participant_id'].unique():
                plt.figure(figsize=(12, 5))
                df_p = df[df['participant_id'] == pid]
                
                sns.lineplot(data=df_p, x='date', y='acwr', label='ACWR')
                
                plt.axhline(y=1.5, color='r', linestyle='--', label='Seuil de Risque (>1.5)')
                
                blessures = df_p[df_p['is_injured'] == 1]
                if not blessures.empty:
                    plt.scatter(blessures['date'], blessures['acwr'], 
                                color='red', label='Blessure', zorder=5, s=80, marker='X')
                
                plt.title(f"Évolution ACWR & Blessures (Participant {pid.upper()})")
                plt.legend()
                plt.tight_layout()
                plt.show()

        cols_focus = [
            'is_injured', 'acwr', 'energy_balance', 'fatigue', 
            'readiness', 'sleep_efficiency', 'resting_bpm', 'calories'
        ]
        cols_present = [c for c in cols_focus if c in df.columns]
        
        if len(cols_present) > 1:
            plt.figure(figsize=(12, 10))
            sns.heatmap(df[cols_present].corr(), annot=True, cmap='RdYlGn', center=0, fmt=".2f")
            plt.title("Matrice de Corrélation (Performance & Risque)")
            plt.show()

    output_path = cfg.DATA_DIR / "enriched_athlete_data.csv"
    df.to_csv(output_path, index=False)

    return df

def analyze_enriched_data(df: pd.DataFrame, plot: bool = True):
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['participant_id', 'date'])

    if 'sleep_efficiency' in df.columns:
        df['sleep_eff_3d_avg'] = df.groupby('participant_id')['sleep_efficiency'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
    
    if 'energy_balance' in df.columns:
        df['energy_bal_3d_avg'] = df.groupby('participant_id')['energy_balance'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )

    blessures = df[df['is_injured'] == 1]
    if not blessures.empty:
        cols_stats = ['acwr', 'sleep_eff_3d_avg', 'energy_bal_3d_avg']
        cols_present = [c for c in cols_stats if c in df.columns]
        
        print("Stats Moyennes (3 jours avant blessure):")
        print(df[df['is_injured'] == 1][cols_present].mean())
    else:
        print("Aucune blessure trouvée pour les stats.")

    key_metrics = [
        'training_load', 'acwr', 'sleep_efficiency', 
        'energy_balance', 'fatigue', 'soreness', 'readiness'
    ]
    metrics_present = [c for c in key_metrics if c in df.columns]

    if metrics_present:
        try:
            comparison = df.groupby('is_injured')[metrics_present].mean().T
            if comparison.shape[1] == 2:
                comparison.columns = ['Moyenne Sains', 'Moyenne Blessés']
                comparison['Différence (%)'] = (
                    (comparison['Moyenne Blessés'] - comparison['Moyenne Sains']) / 
                    comparison['Moyenne Sains']
                ) * 100
            print(comparison.round(2))
        except Exception as e:
            print(f"Erreur calcul comparatif : {e}")

    agg_dict = {'is_injured': 'sum'}
    if 'fatigue' in df.columns: agg_dict['fatigue'] = 'count'
    if 'calories' in df.columns: agg_dict['calories'] = 'count'

    report_check = df.groupby('participant_id').agg(agg_dict).rename(
        columns={'is_injured': 'Nb_Blessures', 'fatigue': 'Rapports_Wellness', 'calories': 'Photos_Nutrition'}
    )
    
    print(report_check.head())

    return df

def visualize_distributions_by_participant(df: pd.DataFrame, specific_participants: list = None):
    
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    n_vars = len(numeric_cols)

    if specific_participants:
        participants = [p for p in specific_participants if p in df['participant_id'].unique()]
    else:
        participants = df['participant_id'].unique()

    for p in participants:
        sub_df = df[df['participant_id'] == p]
        
        n_cols_plot = 4
        n_rows_plot = math.ceil(n_vars / n_cols_plot)
        
        plt.figure(figsize=(20, 3 * n_rows_plot)) 
        plt.suptitle(f"Distributions complètes - Participant {p}", fontsize=16, y=1.002)
        
        for i, col in enumerate(numeric_cols):
            ax = plt.subplot(n_rows_plot, n_cols_plot, i+1)
            
            data_clean = sub_df[col].dropna()
            
            if not data_clean.empty:
                if data_clean.nunique() <= 1:
                    plt.hist(data_clean, color='gray')
                    plt.title(f"{col} (Constante)", fontsize=9, color='red', fontweight='bold')
                
                else:
                    sns.histplot(data_clean, kde=True, color='#2ecc71', edgecolor='k', linewidth=0.5)
                    plt.title(f"{col}", fontsize=9, fontweight='bold')
            else:
                plt.text(0.5, 0.5, "PAS DE DONNÉES", ha='center', va='center', color='red')
                plt.title(f"{col}", fontsize=9)
            
            plt.xlabel("") 
            plt.ylabel("")

        plt.tight_layout()
        plt.show()