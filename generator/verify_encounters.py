
import csv
import sqlite3
from pathlib import Path
from datetime import date

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ENCOUNTER_FILE = (
    PROJECT_ROOT / "data" / "bronze" / "encounters"
    / "encounters_test.csv"
)
FACILITY_FILE = (
    PROJECT_ROOT / "data" / "bronze" / "facilities"
    / "facilities.csv"
)
GROUND_TRUTH_DIR = PROJECT_ROOT / "data" / "ground_truth"

OUTBREAK_FILE = GROUND_TRUTH_DIR / "ground_truth_outbreaks.csv"
SILENCE_FILE = GROUND_TRUTH_DIR / "facility_silence_events.csv"

DB_FILE = (
    PROJECT_ROOT / "data" / "bronze" / "encounters"
    / "encounter_validation_temp.sqlite"
)

REQUIRED_COLUMNS = [
    "encounter_id", "patient_id", "facility_id", "district",
    "onset_date", "report_date", "age_years", "sex",
    "icd10_code", "lab_confirmed", "admitted", "outcome"
]

VALID_SEX = {"Male", "Female"}

VALID_CODES = {
    "A90", "A91", "A00", "A09", "J10", "J11",
    "B50", "B51", "B52", "B53", "B54",
    "Z00", "R50", "B99"
}

VALID_BOOLEAN = {"True", "False", "true", "false", "0", "1"}

BATCH_SIZE = 10000


def read_small_csv(path):
    if not path.exists():
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        rows = list(reader)

    return rows, columns


def main():
    print("=" * 60)
    print("       EPISCOPE LARGE DATASET VALIDATION")
    print("=" * 60)

    if not ENCOUNTER_FILE.exists():
        raise SystemExit(f"Missing encounter file: {ENCOUNTER_FILE}")

    # Load the small facility registry.
    facilities, _ = read_small_csv(FACILITY_FILE)

    valid_facility_ids = {
        row["facility_id"] for row in facilities
    }

    print("\nRegistered facilities:", len(valid_facility_ids))

    # Prepare disk-backed duplicate detection.
    if DB_FILE.exists():
        DB_FILE.unlink()

    conn = sqlite3.connect(str(DB_FILE))
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=FILE")
    conn.execute("PRAGMA cache_size=-64000")

    conn.execute("""
        CREATE TABLE encounter_ids (
            encounter_id TEXT PRIMARY KEY
        ) WITHOUT ROWID
    """)

    counters = {
        "total": 0,
        "duplicate_ids": 0,
        "invalid_ages": 0,
        "invalid_sex": 0,
        "orphans": 0,
        "invalid_dates": 0,
        "negative_delays": 0,
        "delays_above_21": 0,
        "invalid_codes": 0,
        "invalid_booleans": 0,
        "malformed_rows": 0
    }

    sex_counts = {}
    code_counts = {}

    min_delay = None
    max_delay = None

    batch = []

    try:
        print("\nReading encounter CSV...")
        print("This may take some time. Please wait.")

        with open(
            ENCOUNTER_FILE,
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as f:

            reader = csv.DictReader(f)
            actual_columns = reader.fieldnames or []

            missing_columns = [
                c for c in REQUIRED_COLUMNS
                if c not in actual_columns
            ]

            print(
                "\nAll required columns present:",
                not missing_columns
            )

            if missing_columns:
                print("Missing columns:", missing_columns)
                raise SystemExit("Required columns are missing.")

            for row in reader:
                counters["total"] += 1

                if None in row:
                    counters["malformed_rows"] += 1
                    continue

                # Duplicate ID detection batch.
                batch.append((row.get("encounter_id", ""),))

                if len(batch) >= BATCH_SIZE:
                    before = conn.total_changes

                    conn.executemany(
                        "INSERT OR IGNORE INTO encounter_ids "
                        "(encounter_id) VALUES (?)",
                        batch
                    )

                    inserted = conn.total_changes - before
                    counters["duplicate_ids"] += len(batch) - inserted

                    conn.commit()
                    batch.clear()

                # Age validation.
                try:
                    age = int(row["age_years"])
                    if not 0 <= age <= 120:
                        counters["invalid_ages"] += 1
                except (ValueError, TypeError):
                    counters["invalid_ages"] += 1

                # Sex validation.
                sex = row["sex"]
                sex_counts[sex] = sex_counts.get(sex, 0) + 1

                if sex not in VALID_SEX:
                    counters["invalid_sex"] += 1

                # Facility validation.
                if row["facility_id"] not in valid_facility_ids:
                    counters["orphans"] += 1

                # Date and delay validation.
                try:
                    onset = date.fromisoformat(row["onset_date"])
                    report = date.fromisoformat(row["report_date"])

                    delay = (report - onset).days

                    if min_delay is None or delay < min_delay:
                        min_delay = delay

                    if max_delay is None or delay > max_delay:
                        max_delay = delay

                    if delay < 0:
                        counters["negative_delays"] += 1

                    if delay > 21:
                        counters["delays_above_21"] += 1

                except (ValueError, TypeError):
                    counters["invalid_dates"] += 1

                # ICD-10 validation.
                code = row["icd10_code"]
                code_counts[code] = code_counts.get(code, 0) + 1

                if code not in VALID_CODES:
                    counters["invalid_codes"] += 1

                # Boolean validation.
                if (
                    row["lab_confirmed"] not in VALID_BOOLEAN
                    or row["admitted"] not in VALID_BOOLEAN
                ):
                    counters["invalid_booleans"] += 1

                # Progress.
                if counters["total"] % 1_000_000 == 0:
                    print(
                        f"Processed {counters['total']:,} rows..."
                    )

            # Process remaining duplicate IDs.
            if batch:
                before = conn.total_changes

                conn.executemany(
                    "INSERT OR IGNORE INTO encounter_ids "
                    "(encounter_id) VALUES (?)",
                    batch
                )

                inserted = conn.total_changes - before
                counters["duplicate_ids"] += len(batch) - inserted

                conn.commit()

        unique_ids = conn.execute(
            "SELECT COUNT(*) FROM encounter_ids"
        ).fetchone()[0]

        # Ground-truth validation.
        outbreaks, outbreak_columns = read_small_csv(OUTBREAK_FILE)
        silence_events, silence_columns = read_small_csv(SILENCE_FILE)

        required_outbreak_columns = {
            "outbreak_id", "disease", "district",
            "start_date", "end_date", "severity", "outbreak_type"
        }

        outbreak_columns_valid = required_outbreak_columns.issubset(
            set(outbreak_columns)
        )

        outbreak_ids = [
            row["outbreak_id"] for row in outbreaks
        ]

        duplicate_outbreak_ids = (
            len(outbreak_ids) - len(set(outbreak_ids))
        )

        required_silence_columns = {
            "facility_id", "district", "silence_start", "silence_end"
        }

        silence_columns_valid = required_silence_columns.issubset(
            set(silence_columns)
        )

        # Results.
        print("\n" + "=" * 60)
        print("              VALIDATION RESULTS")
        print("=" * 60)

        print("\n1. BASIC VALIDATION")
        print("Total encounter rows:", f"{counters['total']:,}")
        print("Unique encounter IDs:", f"{unique_ids:,}")
        print("Malformed rows:", counters["malformed_rows"])

        print("\n2. DUPLICATE VALIDATION")
        print("Duplicate encounter IDs:",
              f"{counters['duplicate_ids']:,}")

        print("\n3. AGE VALIDATION")
        print("Invalid age records:",
              f"{counters['invalid_ages']:,}")

        print("\n4. SEX VALIDATION")
        print("Invalid sex records:",
              f"{counters['invalid_sex']:,}")
        print("Sex distribution:", sex_counts)

        print("\n5. FACILITY VALIDATION")
        print("Registered facilities:", len(valid_facility_ids))
        print("Orphan facility records:",
              f"{counters['orphans']:,}")

        print("\n6. DATE VALIDATION")
        print("Invalid date records:", counters["invalid_dates"])
        print("Negative reporting delays:",
              counters["negative_delays"])
        print("Delays above 21 days:",
              counters["delays_above_21"])
        print("Minimum delay:", min_delay)
        print("Maximum delay:", max_delay)

        print("\n7. ICD-10 VALIDATION")
        print("Invalid ICD-10 records:",
              counters["invalid_codes"])
        print("ICD-10 distribution:", code_counts)

        print("\n8. BOOLEAN VALIDATION")
        print("Invalid boolean records:",
              counters["invalid_booleans"])

        print("\n9. GROUND-TRUTH VALIDATION")
        print("Outbreak events:", len(outbreaks))
        print("Facility silence events:", len(silence_events))
        print("Outbreak columns valid:", outbreak_columns_valid)
        print("Duplicate outbreak IDs:", duplicate_outbreak_ids)
        print("Facility silence columns valid:",
              silence_columns_valid)

        # Final checks.
        checks = {
            "Required columns": not missing_columns,
            "Age defects present": counters["invalid_ages"] > 0,
            "Invalid sex defects present": counters["invalid_sex"] > 0,
            "Duplicate submissions present":
                counters["duplicate_ids"] > 0,
            "Orphan facility defects present":
                counters["orphans"] > 0,
            "No negative reporting delays":
                counters["negative_delays"] == 0,
            "Reporting delays within 21 days":
                counters["delays_above_21"] == 0,
            "Valid dates": counters["invalid_dates"] == 0,
            "Valid ICD-10 codes": counters["invalid_codes"] == 0,
            "Valid boolean fields":
                counters["invalid_booleans"] == 0,
            "No malformed rows": counters["malformed_rows"] == 0,
            "At least 30 outbreak events": len(outbreaks) >= 30,
            "Valid outbreak columns": outbreak_columns_valid,
            "No duplicate outbreak IDs":
                duplicate_outbreak_ids == 0,
            "Facility silence events present":
                len(silence_events) > 0,
            "Valid facility silence columns":
                silence_columns_valid
        }

        print("\n10. FINAL SUMMARY")

        for name, passed in checks.items():
            print(
                f"{'PASS' if passed else 'FAIL'}: {name}"
            )

        print("\n" + "=" * 60)

        if all(checks.values()):
            print("ENCOUNTER VALIDATION PASSED!")
        else:
            print("VALIDATION NEEDS ATTENTION.")

        print("=" * 60)

    finally:
        conn.close()

        if DB_FILE.exists():
            DB_FILE.unlink()

        print("\nTemporary validation database cleaned up.")


if __name__ == "__main__":
    main()
