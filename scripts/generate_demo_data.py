# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pandas",
#     "numpy",
# ]
# ///
"""
Generate synthetic ACCIDENT data for the speed camera analysis marimo notebook.

Camera locations are REAL public data from Pingtung County.
Only the accident records are synthetic — the data structure mirrors real
government datasets but all accident values are entirely fictional.

Usage:
    uv run scripts/generate_demo_data.py
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

# --- Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# === 1. Load REAL camera data ===
cam_path = os.path.join(DATA_DIR, "cameras.csv")
df_cameras = pd.read_csv(cam_path)

lon_col = [c for c in df_cameras.columns if "Longitude" in c][0]
lat_col = [c for c in df_cameras.columns if "Latitude" in c][0]
date_col = [c for c in df_cameras.columns if c in ["設置日期"] or "日期" in c]
date_col = date_col[0] if date_col else df_cameras.columns[3]

print(f"Loaded {len(df_cameras)} real cameras from {cam_path}")

# === 2. Generate Synthetic Accident Data ===
ACCIDENT_TYPES = [12, 13, 14, 18, 19, 20, 22, 23, 24, 25, 27, 28]
# Pre-2023/07 cause codes
CAUSE_CODES_OLD = [13, 14, 16, 17]
# Post-2023/07 cause codes
CAUSE_CODES_NEW = [5, 6, 7, 8]


def roc_date_int(year, month, day):
    """Convert to ROC date integer: YYYMMDD"""
    roc_year = year - 1911
    return roc_year * 10000 + month * 100 + day


def parse_setup_year_month(val):
    """Parse setup date string like '109.05' to (year, month)."""
    try:
        val_str = str(val)
        if "." in val_str:
            parts = val_str.split(".")
            return int(parts[0]) + 1911, int(parts[1])
        else:
            # Try as YYYMMDD integer
            s = str(int(float(val_str)))
            if len(s) == 7:
                return int(s[:3]) + 1911, int(s[3:5])
            elif len(s) == 6:
                return int(s[:2]) + 1911, int(s[2:4])
    except:
        pass
    return 2020, 1  # fallback


accidents = []
acc_id_counter = 100000

# 2a. For each camera, generate accidents WITHIN buffer (~120m offset)
for _, cam in df_cameras.iterrows():
    cam_lon = cam[lon_col]
    cam_lat = cam[lat_col]

    if pd.isna(cam_lon) or pd.isna(cam_lat):
        continue

    setup_year, setup_month = parse_setup_year_month(cam[date_col])

    # Generate 3~8 accidents near this camera
    n_acc = np.random.randint(3, 9)
    for _ in range(n_acc):
        # Random offset within ~120m (approx 0.001 degrees at this latitude)
        offset_lon = np.random.uniform(-0.001, 0.001)
        offset_lat = np.random.uniform(-0.001, 0.001)

        # Random date: 36 months before to 36 months after setup
        month_offset = np.random.randint(-36, 37)
        acc_year = setup_year + (setup_month + month_offset - 1) // 12
        acc_month = (setup_month + month_offset - 1) % 12 + 1
        acc_day = np.random.randint(1, 29)

        # Clamp to reasonable range
        acc_year = max(2016, min(2025, acc_year))

        # Determine cause code based on date (code system changed 2023/07)
        if acc_year > 2023 or (acc_year == 2023 and acc_month >= 7):
            cause = np.random.choice(CAUSE_CODES_NEW)
        else:
            cause = np.random.choice(CAUSE_CODES_OLD)

        acc_id_counter += 1
        accidents.append({
            "總編號": f"DEMO{acc_id_counter}",
            "發生日期": roc_date_int(acc_year, acc_month, acc_day),
            "GPS經度": round(cam_lon + offset_lon, 6),
            "GPS緯度": round(cam_lat + offset_lat, 6),
            "道路類別": 2,  # Provincial road
            "事故類型及型態": np.random.choice(ACCIDENT_TYPES),
            "主要肇事因素": cause,
            "死亡": np.random.choice([0, 0, 0, 0, 0, 1], p=[0.5, 0.2, 0.1, 0.1, 0.05, 0.05]),
            "受傷": np.random.randint(0, 4),
            "發生年度": acc_year,
            "發生月份": acc_month,
        })

# 2b. Background accidents (not near any camera, for control group)
# Use Pingtung County bounding box
LAT_MIN, LAT_MAX = 21.9, 22.8
LON_MIN, LON_MAX = 120.3, 120.9
N_BACKGROUND = 500

for _ in range(N_BACKGROUND):
    acc_year = np.random.randint(2016, 2026)
    acc_month = np.random.randint(1, 13)
    acc_day = np.random.randint(1, 29)

    if acc_year > 2023 or (acc_year == 2023 and acc_month >= 7):
        cause = np.random.choice(CAUSE_CODES_NEW)
    else:
        cause = np.random.choice(CAUSE_CODES_OLD)

    acc_id_counter += 1
    accidents.append({
        "總編號": f"DEMO{acc_id_counter}",
        "發生日期": roc_date_int(acc_year, acc_month, acc_day),
        "GPS經度": round(np.random.uniform(LON_MIN, LON_MAX), 6),
        "GPS緯度": round(np.random.uniform(LAT_MIN, LAT_MAX), 6),
        "道路類別": 2,
        "事故類型及型態": np.random.choice(ACCIDENT_TYPES),
        "主要肇事因素": cause,
        "死亡": np.random.choice([0, 0, 0, 0, 0, 1], p=[0.5, 0.2, 0.1, 0.1, 0.05, 0.05]),
        "受傷": np.random.randint(0, 4),
        "發生年度": acc_year,
        "發生月份": acc_month,
    })

df_accidents = pd.DataFrame(accidents)

# === Save ===
acc_out = os.path.join(DATA_DIR, "accidents_demo.csv")
df_accidents.to_csv(acc_out, index=False, encoding="utf-8-sig")

n_near = len(accidents) - N_BACKGROUND
print(f"[OK] Generated {len(df_accidents)} synthetic accidents -> {acc_out}")
print(f"     ({n_near} near real cameras + {N_BACKGROUND} background)")
