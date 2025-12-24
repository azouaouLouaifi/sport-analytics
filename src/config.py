import os
from pathlib import Path

PROJECT_ROOT = Path("/app") 
DATA_DIR = PROJECT_ROOT / "data"

HR_ZONE_MAPPING = {
    "BELOW_DEFAULT_ZONE_1": "zone_0_rest",
    "IN_DEFAULT_ZONE_1": "zone_1_fat_burn",   
    "IN_DEFAULT_ZONE_2": "zone_2_cardio",     
    "IN_DEFAULT_ZONE_3": "zone_3_peak"        
}

PARTICIPANTS = ["p01", "p03", "p05"]