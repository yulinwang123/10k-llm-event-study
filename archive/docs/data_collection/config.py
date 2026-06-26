"""
config.py — central settings for the earnings attention project
Edit WRDS_USERNAME before running any script.
"""

# ── WRDS ────────────────────────────────────────────────────────────────────
WRDS_USERNAME = "your_wrds_username"   # ← change this

# ── Sample period ────────────────────────────────────────────────────────────
START_DATE = "2010-01-01"
END_DATE   = "2023-12-31"

# ── Output paths ─────────────────────────────────────────────────────────────
import os
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")

IBES_FILE        = os.path.join(DATA_DIR, "ibes_quarterly.parquet")
LINK_FILE        = os.path.join(DATA_DIR, "ibes_crsp_link.parquet")
CRSP_RAW_FILE    = os.path.join(DATA_DIR, "crsp_daily_raw.parquet")
CAR_FILE         = os.path.join(DATA_DIR, "car_panel.parquet")
TRANSCRIPT_FILE  = os.path.join(DATA_DIR, "transcripts.parquet")
MASTER_FILE      = os.path.join(DATA_DIR, "master_panel.parquet")
MASTER_CSV       = os.path.join(DATA_DIR, "master_panel.csv")

os.makedirs(DATA_DIR, exist_ok=True)

# ── Event windows ─────────────────────────────────────────────────────────────
# CAR_SHORT: [-1, +1] around announcement date (immediate reaction)
# CAR_LONG : [+2, +60] post-announcement drift window
CAR_SHORT_WINDOW = (-1, 1)
CAR_LONG_WINDOW  = (2, 60)

# Extra buffer of trading days to pull from CRSP around each event
CRSP_BUFFER_DAYS = 90   # calendar days on each side of announcement
