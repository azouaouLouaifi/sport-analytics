import json
import ast
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import src.config as cfg

class FitbitDataLoader:
    """
    Charge et normalise TOUS les fichiers Fitbit (Steps, Heart Rate, Sleep, Exercises...).
    """

    def __init__(self, participant_id: str):
        self.p_id = participant_id
        self.base_path = cfg.DATA_DIR / participant_id / "fitbit"

    def load_json(self, filename: str) -> Union[List, Dict, None]:
        file_path = self.base_path / filename
        if not file_path.exists():
            print(f"Fichier introuvable : {file_path}")
            return None
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Erreur lecture {filename}: {e}")
            return None

    def load_simple_timeseries(self, filename: str) -> pd.DataFrame:
        data = self.load_json(filename)
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df['dateTime'] = pd.to_datetime(df['dateTime'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        return df

    def load_heart_rate(self) -> pd.DataFrame:
        data = self.load_json("heart_rate.json")
        if not data:
            return pd.DataFrame()

        processed_data = [
            {
                "dateTime": entry["dateTime"],
                "bpm": entry["value"]["bpm"],
                "confidence": entry["value"].get("confidence", 0)
            }
            for entry in data
        ]
        
        df = pd.DataFrame(processed_data)
        df['dateTime'] = pd.to_datetime(df['dateTime'])
        return df
    
    def load_resting_heart_rate(self) -> pd.DataFrame:
        data = self.load_json("resting_heart_rate.json")
        if not data:
            return pd.DataFrame()

        processed_data = []
        for entry in data:
            val_obj = entry.get("value")
            
            bpm_val = None
            error_val = 0.0
            
            if isinstance(val_obj, dict):
                bpm_val = val_obj.get("value")
                error_val = val_obj.get("error", 0.0)
            else:
                bpm_val = val_obj
            
            if bpm_val is not None:
                processed_data.append({
                    "dateTime": pd.to_datetime(entry["dateTime"]),
                    "resting_bpm": float(bpm_val),
                    "bpm_error": float(error_val)
                })
            
        df = pd.DataFrame(processed_data)
        
        if not df.empty:
            df = df.sort_values("dateTime").reset_index(drop=True)
            
        return df

    def load_heart_rate_zones(self) -> pd.DataFrame:
        data = self.load_json("time_in_heart_rate_zones.json")
        if not data:
            return pd.DataFrame()

        processed_data = []
        for entry in data:
            zones = entry.get("value", {}).get("valuesInZones", {})
            row = {
                "dateTime": pd.to_datetime(entry["dateTime"]),
                "minutes_below_zone": zones.get("BELOW_DEFAULT_ZONE_1", 0),
                "minutes_fat_burn": zones.get("IN_DEFAULT_ZONE_1", 0),
                "minutes_cardio": zones.get("IN_DEFAULT_ZONE_2", 0),
                "minutes_peak": zones.get("IN_DEFAULT_ZONE_3", 0)
            }
            processed_data.append(row)

        return pd.DataFrame(processed_data)

    def load_sleep(self) -> pd.DataFrame:
        data = self.load_json("sleep.json")
        if not data:
            return pd.DataFrame()

        clean_rows = []
        for entry in data:
            row = {
                "dateOfSleep": entry.get("dateOfSleep"),
                "startTime": entry.get("startTime"),
                "endTime": entry.get("endTime"),
                "duration_min": entry.get("duration", 0) / 60000, 
                "timeInBed_min": entry.get("timeInBed"),
                "efficiency": entry.get("efficiency"),
                "isMainSleep": entry.get("mainSleep", False)
            }
            
            summary = entry.get("levels", {}).get("summary", {})
            
            for stage in ["deep", "light", "rem", "wake"]:
                stage_data = summary.get(stage, {})
                row[f"minutes_{stage}"] = stage_data.get("minutes", 0)
                row[f"count_{stage}"] = stage_data.get("count", 0)

            clean_rows.append(row)

        df = pd.DataFrame(clean_rows)
        
        if not df.empty:
            for col in ["dateOfSleep", "startTime", "endTime"]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
                    
            df = df.sort_values("startTime").reset_index(drop=True)

        return df

    def load_sleep_score_csv(self) -> pd.DataFrame:
        file_path = self.base_path / "sleep_score.csv"
        if not file_path.exists():
            return pd.DataFrame()
        
        df = pd.read_csv(file_path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def load_exercises(self) -> pd.DataFrame:
        data = self.load_json("exercise.json")
        if not data:
            return pd.DataFrame()
        
        flat_data = []
        
        for entry in data:
            row = {
                "logId": entry.get("logId"),
                "activityName": entry.get("activityName"),
                "startTime": entry.get("startTime"),
                "duration_ms": entry.get("duration"),
                "calories": entry.get("calories"),
                "steps": entry.get("steps", 0),
                "avg_heart_rate": entry.get("averageHeartRate"),
                "elevationGain": entry.get("elevationGain", 0)
            }

            levels = entry.get("activityLevel", [])
            for lvl in levels:
                name = lvl.get("name", "unknown")
                minutes = lvl.get("minutes", 0)
                row[f"minutes_level_{name}"] = minutes

            zones = entry.get("heartRateZones", [])
            for zone in zones:
                raw_name = zone.get("name", "unknown")
                clean_name = raw_name.replace(" ", "_").lower()
                minutes = zone.get("minutes", 0)
                row[f"minutes_zone_{clean_name}"] = minutes
            
            flat_data.append(row)
        
        df = pd.DataFrame(flat_data)
        
        if "startTime" in df.columns:
            df["startTime"] = pd.to_datetime(df["startTime"])
            
        return df


class SecondaryDataLoader:
 
    def __init__(self, participant_id: str):
        self.p_id = participant_id
        self.pmsys_path = cfg.DATA_DIR / participant_id / "pmsys"
        self.gdocs_path = cfg.DATA_DIR / participant_id / "googledocs"
        self.overview_path = cfg.DATA_DIR / "participant-overview.xlsx"

    def _load_csv(self, folder, filename):
        path = folder / filename
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)

    def load_wellness(self) -> pd.DataFrame:
        df = self._load_csv(self.pmsys_path, "wellness.csv")
        if df.empty: return df

        df['date'] = pd.to_datetime(df['effective_time_frame']).dt.normalize()
        
        cols_to_avg = ['fatigue', 'mood', 'readiness', 'sleep_duration_h', 'sleep_quality', 'stress', 'soreness']
        df_agg = df.groupby('date')[cols_to_avg].mean().reset_index()
        
        return df_agg

    def load_srpe(self) -> pd.DataFrame:
        df = self._load_csv(self.pmsys_path, "srpe.csv")
        if df.empty: return df

        df['date'] = pd.to_datetime(df['end_date_time']).dt.normalize()
        
        df['training_load'] = df['perceived_exertion'] * df['duration_min']
        
        df_agg = df.groupby('date')[['training_load', 'duration_min']].sum().reset_index()
        df_agg.rename(columns={'duration_min': 'training_duration_min'}, inplace=True)
        
        return df_agg

    def load_injury(self) -> pd.DataFrame:
        df = self._load_csv(self.pmsys_path, "injury.csv")
        if df.empty: return df

        df['date'] = pd.to_datetime(df['effective_time_frame']).dt.normalize()
        
        def parse_injury(val):
            try:
                d = ast.literal_eval(val) 
                return 1 if len(d) > 0 else 0
            except:
                return 0

        df['is_injured'] = df['injuries'].apply(parse_injury)
        return df[['date', 'is_injured']]

    def load_reporting(self) -> pd.DataFrame:
        df = self._load_csv(self.gdocs_path, "reporting.csv")
        if df.empty: return df

        df['date'] = pd.to_datetime(df['date'], format="%d/%m/%Y", errors='coerce')
        
        if 'alcohol_consumed' in df.columns:
            df['alcohol_consumed'] = df['alcohol_consumed'].map({'Yes': 1, 'No': 0}).fillna(0)
            
        return df[['date', 'weight', 'glasses_of_fluid', 'alcohol_consumed']]

    def load_overview(self) -> dict:
        
        if not self.overview_path.exists():
            return {}
        
        try:
            df = pd.read_excel(self.overview_path, sheet_name=0, header=1)
        except Exception as e:
            print(f"Erreur lecture Excel : {e}")
            return {}

        df.columns = df.columns.str.strip()
        
        row = df[df['Participant ID'] == self.p_id]
        if row.empty: return {}
        
        data = row.iloc[0].to_dict()
        
        final_data = {}
        
        for key in ['Age', 'Height', 'Gender', 'Max heart rate']:
            if key in data:
                final_data[key] = data[key]

        if 'Stride walk' in data:
            final_data['stride_walk_cm'] = data['Stride walk']
        if 'Stride run' in data:
            final_data['stride_run_cm'] = data['Stride run']

        mins = data.get('Minutes')
        secs = data.get('Seconds')
        
        def clean_num(val):
            try:
                return float(val)
            except (ValueError, TypeError):
                return None

        mins_clean = clean_num(mins)
        secs_clean = clean_num(secs)

        if mins_clean is not None and secs_clean is not None:
            total_time_min = mins_clean + (secs_clean / 60.0)
            final_data['benchmark_5km_time_min'] = round(total_time_min, 2)
            
            if total_time_min > 0:
                speed_kmh = 5 / (total_time_min / 60.0)
                final_data['benchmark_5km_speed_kmh'] = round(speed_kmh, 2)
        else:
            final_data['benchmark_5km_time_min'] = None

        return final_data


class DataAggregator:
    def __init__(self, fitbit_data: dict, secondary_data: dict, static_data: dict):
        self.fitbit = fitbit_data
        self.secondary = secondary_data
        self.static = static_data

    def run(self) -> pd.DataFrame:
        """Fusionne toutes les sources en un seul DataFrame chronologique."""
        
        if self.fitbit.get('steps') is None or self.fitbit['steps'].empty:
            return pd.DataFrame()

        df_base = self.fitbit['steps'].groupby(self.fitbit['steps']['dateTime'].dt.normalize())['value'].sum().reset_index()
        df_base.columns = ['date', 'total_steps']
        
        df_base['date'] = pd.to_datetime(df_base['date']).dt.tz_localize(None)

        if not self.fitbit['calories'].empty:
            cal = self.fitbit['calories'].groupby(self.fitbit['calories']['dateTime'].dt.normalize())['value'].sum().reset_index()
            cal.columns = ['date', 'total_calories']
            cal['date'] = pd.to_datetime(cal['date']).dt.tz_localize(None)
            df_base = pd.merge(df_base, cal, on='date', how='outer')

        if not self.fitbit['distance'].empty:
            dist = self.fitbit['distance'].groupby(self.fitbit['distance']['dateTime'].dt.normalize())['value'].sum().reset_index()
            dist.columns = ['date', 'total_distance_cm']
            dist['date'] = pd.to_datetime(dist['date']).dt.tz_localize(None)
            df_base = pd.merge(df_base, dist, on='date', how='left')

        if not self.fitbit['resting_heart_rate'].empty:
            rhr = self.fitbit['resting_heart_rate'][['dateTime', 'resting_bpm', 'bpm_error']].copy()
            rhr['date'] = rhr['dateTime'].dt.normalize().dt.tz_localize(None)
            df_base = pd.merge(df_base, rhr.drop(columns=['dateTime']), on='date', how='left')

        if 'hr_zones' in self.fitbit and not self.fitbit['hr_zones'].empty:
            hr_z = self.fitbit['hr_zones'].copy()
            hr_z['date'] = hr_z['dateTime'].dt.normalize().dt.tz_localize(None)
            zones_agg = hr_z.groupby('date')[['minutes_below_zone', 'minutes_fat_burn', 'minutes_cardio', 'minutes_peak']].sum().reset_index()
            df_base = pd.merge(df_base, zones_agg, on='date', how='left')

        if not self.fitbit['sleep'].empty:
            sleep = self.fitbit['sleep'].copy()
            sleep['date'] = sleep['startTime'].dt.normalize().dt.tz_localize(None)
            
            agg_rules = {
                'duration_min': 'sum', 'efficiency': 'mean',
                'minutes_deep': 'sum', 'minutes_rem': 'sum',
                'minutes_light': 'sum', 'minutes_wake': 'sum',
                'count_wake': 'sum', 'timeInBed_min': 'sum'
            }
            agg_rules = {k: v for k, v in agg_rules.items() if k in sleep.columns}
            sleep_daily = sleep.groupby('date').agg(agg_rules).reset_index().add_prefix('sleep_')
            sleep_daily = sleep_daily.rename(columns={'sleep_date': 'date'})
            
            df_base = pd.merge(df_base, sleep_daily, on='date', how='left')

        if 'sleep_score' in self.fitbit and not self.fitbit['sleep_score'].empty:
            ss = self.fitbit['sleep_score'].copy()
            if 'timestamp' in ss.columns:
                ss['date'] = ss['timestamp'].dt.normalize().dt.tz_localize(None)
                cols_to_keep = ['overall_score', 'composition_score', 'revitalization_score', 'restlessness_score']
                cols = [c for c in cols_to_keep if c in ss.columns]
                if cols:
                    ss_agg = ss.groupby('date')[cols].mean().reset_index()
                    df_base = pd.merge(df_base, ss_agg, on='date', how='left')

        if 'exercises' in self.fitbit and not self.fitbit['exercises'].empty:
            ex = self.fitbit['exercises'].copy()
            ex['date'] = ex['startTime'].dt.normalize().dt.tz_localize(None)
            
            ex_agg = ex.groupby('date').agg({
                'duration_ms': 'sum',
                'calories': 'sum',
                'logId': 'count'
            }).reset_index()
            ex_agg.columns = ['date', 'sport_duration_ms', 'sport_calories', 'sport_sessions_count']
            ex_agg['sport_duration_min'] = ex_agg['sport_duration_ms'] / 60000
            ex_agg = ex_agg.drop(columns=['sport_duration_ms'])
            
            df_base = pd.merge(df_base, ex_agg, on='date', how='left')

        def prepare_secondary(df_sec):
            if df_sec.empty: return df_sec
            df_clean = df_sec.copy()
            df_clean['date'] = pd.to_datetime(df_clean['date']).dt.tz_localize(None)
            return df_clean

        if not self.secondary['wellness'].empty:
            df_well = prepare_secondary(self.secondary['wellness'])
            df_base = pd.merge(df_base, df_well, on='date', how='left')

        if not self.secondary['srpe'].empty:
            df_srpe = prepare_secondary(self.secondary['srpe'])
            df_base = pd.merge(df_base, df_srpe, on='date', how='left')
            df_base['training_load'] = df_base['training_load'].fillna(0)

        df_base['is_injured'] = 0
        if not self.secondary['injury'].empty:
            df_inj = prepare_secondary(self.secondary['injury'])
            inj_dates = df_inj[df_inj['is_injured'] == 1]['date']
            df_base.loc[df_base['date'].isin(inj_dates), 'is_injured'] = 1

        if not self.secondary['reporting'].empty:
            df_rep = prepare_secondary(self.secondary['reporting'])
            df_base = pd.merge(df_base, df_rep, on='date', how='left')
            
            df_base['weight'] = df_base['weight'].ffill()
            if 'glasses_of_fluid' in df_base.columns:
                df_base['glasses_of_fluid'] = df_base['glasses_of_fluid'].fillna(df_base['glasses_of_fluid'].median())

        for col, val in self.static.items():
            if col not in ['Participant ID', '1st 5km run']:
                df_base[col] = val

        df_base = df_base.sort_values('date').reset_index(drop=True)
        return df_base