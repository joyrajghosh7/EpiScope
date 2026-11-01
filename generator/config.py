from pathlib import Path
from schema import START_YEAR, END_YEAR, DEFAULT_SEED, DEFAULT_SCALE_FACTOR

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"

# Generator settings
SEED = DEFAULT_SEED
SCALE_FACTOR = DEFAULT_SCALE_FACTOR

# Date range
START_DATE = f"{START_YEAR}-01-01"
END_DATE = f"{END_YEAR}-12-31"

# Chunk size: generate records in batches to avoid memory problems
CHUNK_SIZE = 100_000

# Output directories
ENCOUNTERS_DIR = BRONZE_DIR / "encounters"
FACILITIES_DIR = BRONZE_DIR / "facilities"
STOCK_DIR = BRONZE_DIR / "stock"
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"


def create_directories():
    directories = [
        ENCOUNTERS_DIR,
        FACILITIES_DIR,
        STOCK_DIR,
        GROUND_TRUTH_DIR,
        SILVER_DIR,
        GOLD_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    create_directories()
    print("EpiScope directories created successfully!")
    print("Scale factor:", SCALE_FACTOR)
    print("Chunk size:", CHUNK_SIZE)
    print("Date range:", START_DATE, "to", END_DATE)