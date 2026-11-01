
import random
import re
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd


# ============================================================
# EPISCOPE - SYNTHETIC ENCOUNTER GENERATOR
# Gamma-Poisson / Negative Binomial district-day model
# ============================================================

SEED = 42
TEST_RECORDS = 100_000

START_DATE = date(2021, 1, 1)
END_DATE = date(2025, 12, 31)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

FACILITIES_FILE = DATA_DIR / "bronze" / "facilities" / "facilities.csv"
POPULATION_FILE = DATA_DIR / "reference" / "district_population.csv"
POPULATION_TOTALS_FILE = DATA_DIR / "reference" / "district_population_totals.csv"
WEATHER_FILE = DATA_DIR / "bronze" / "weather" / "weather_daily.csv"

ENCOUNTER_DIR = DATA_DIR / "bronze" / "encounters"
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"

ENCOUNTER_FILE = ENCOUNTER_DIR / "encounters_test.csv"
OUTBREAK_FILE = GROUND_TRUTH_DIR / "ground_truth_outbreaks.csv"
SILENCE_FILE = GROUND_TRUTH_DIR / "facility_silence_events.csv"

ENCOUNTER_COLUMNS = [
    "encounter_id",
    "patient_id",
    "facility_id",
    "district",
    "onset_date",
    "report_date",
    "age_years",
    "sex",
    "icd10_code",
    "lab_confirmed",
    "admitted",
    "outcome",
]

DISEASES = ["Dengue", "Diarrhoea", "Influenza", "Malaria", "Other"]

DISEASE_CODES = {
    "Dengue": ["A90", "A91"],
    "Diarrhoea": ["A00", "A09"],
    "Influenza": ["J10", "J11"],
    "Malaria": ["B50", "B51", "B52", "B53", "B54"],
    "Other": ["Z00", "R50", "B99"],
}

SEVERITY_PEAK = {
    "Moderate": 4.0,
    "High": 6.0,
    "Critical": 8.0,
}

rng = np.random.default_rng(SEED)
random.seed(SEED)


# ============================================================
# HELPERS
# ============================================================

def find_column(df, possibilities):
    lookup = {str(c).strip().lower(): c for c in df.columns}

    for name in possibilities:
        if name.lower() in lookup:
            return lookup[name.lower()]

    raise ValueError(
        f"Could not find any of {possibilities}. "
        f"Available columns: {list(df.columns)}"
    )


def parse_age_range(label):
    text = str(label).lower().strip()
    numbers = [int(x) for x in re.findall(r"\d+", text)]

    if not numbers:
        return None

    low = numbers[0]

    if "to" in text and len(numbers) >= 2:
        high = numbers[1]
    else:
        high = low + 5

    # Avoid overlapping 85+ and 90+ groups.
    if low >= 85:
        return None

    high = min(high, 84)

    if high < low:
        return None

    return low, high


def date_range(start, end):
    return [
        start + timedelta(days=i)
        for i in range((end - start).days + 1)
    ]


def ensure_directories():
    ENCOUNTER_DIR.mkdir(parents=True, exist_ok=True)
    GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD FACILITIES
# ============================================================

def load_facilities():
    print("Loading facilities...")

    df = pd.read_csv(
        FACILITIES_FILE,
        dtype={"district_code": str}
    )

    code_col = find_column(df, ["district_code"])
    district_col = find_column(df, ["district"])
    facility_col = find_column(df, ["facility_id"])

    facilities = {}

    for _, row in df.iterrows():
        code = str(row[code_col])

        facilities.setdefault(code, []).append({
            "facility_id": str(row[facility_col]),
            "district": str(row[district_col]),
        })

    print(f"Facilities loaded: {len(df)}")
    print(f"Districts covered: {len(facilities)}")

    return facilities


# ============================================================
# LOAD WORLDPOP POPULATION
# ============================================================

def load_population():
    print("Loading WorldPop population data...")

    df = pd.read_csv(
        POPULATION_FILE,
        dtype={"district_code": str}
    )

    code_col = find_column(df, ["district_code"])
    label_col = find_column(
        df,
        ["age_group", "age_label", "age_class"]
    )

    # These names match the actual user's WorldPop CSV.
    male_col = find_column(
        df,
        ["male_population", "male"]
    )

    female_col = find_column(
        df,
        ["female_population", "female"]
    )

    age_profiles = {}
    district_population = {}

    for _, row in df.iterrows():
        code = str(row[code_col])
        age_range = parse_age_range(row[label_col])

        if age_range is None:
            continue

        low, high = age_range

        male = max(0.0, float(row[male_col]))
        female = max(0.0, float(row[female_col]))

        profile = age_profiles.setdefault(code, [])

        if male > 0:
            profile.append((low, high, "Male", male))

        if female > 0:
            profile.append((low, high, "Female", female))

    # Sum the non-overlapping age groups for profile weights.
    for code, profile in age_profiles.items():
        district_population[code] = sum(
            item[3] for item in profile
        )

    # Use official WorldPop district totals for district weights
    # when the separate totals file is available.
    if POPULATION_TOTALS_FILE.exists():
        totals = pd.read_csv(
            POPULATION_TOTALS_FILE,
            dtype={"district_code": str}
        )

        total_code_col = find_column(
            totals,
            ["district_code"]
        )

        total_pop_col = find_column(
            totals,
            ["total_population", "population_total", "population"]
        )

        for _, row in totals.iterrows():
            code = str(row[total_code_col])
            population = float(row[total_pop_col])

            if population > 0:
                district_population[code] = population

    if not age_profiles:
        raise ValueError("No usable WorldPop age-sex profiles found.")

    print(f"District age-sex profiles: {len(age_profiles)}")

    return age_profiles, district_population


# ============================================================
# LOAD WEATHER
# ============================================================

def load_weather():
    print("Loading historical weather...")

    df = pd.read_csv(
        WEATHER_FILE,
        dtype={"district_code": str},
        parse_dates=["date"]
    )

    code_col = find_column(df, ["district_code"])
    date_col = find_column(df, ["date"])
    rain_col = find_column(df, ["precipitation_mm"])
    temp_col = find_column(df, ["temperature_mean_c"])

    weather = {}

    for _, row in df.iterrows():
        code = str(row[code_col])
        day = row[date_col].date()

        weather[(code, day)] = (
            float(row[rain_col]),
            float(row[temp_col])
        )

    print(f"Weather observations loaded: {len(weather)}")

    return weather


# ============================================================
# GENERATE OUTBREAK GROUND TRUTH
# ============================================================

def generate_outbreaks(districts):
    print("Generating outbreak ground truth...")

    events = []

    local_diseases = [
        "Dengue",
        "Diarrhoea",
        "Influenza",
        "Malaria",
    ]

    for i in range(34):
        district = random.choice(districts)
        disease = random.choice(local_diseases)

        start = random_date(
            START_DATE + timedelta(days=30),
            END_DATE - timedelta(days=60)
        )

        duration = random.randint(10, 45)
        end = start + timedelta(days=duration - 1)

        severity = random.choice(
            ["Moderate", "High", "Critical"]
        )

        outbreak_type = (
            "slow_growth" if i == 0 else "regional"
        )

        events.append({
            "outbreak_id": f"OB-{i + 1:03d}",
            "disease": disease,
            "district": district,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "severity": severity,
            "outbreak_type": outbreak_type,
        })

    # Nationwide respiratory surge.
    # It is represented by the Influenza category in this schema.
    events.append({
        "outbreak_id": "OB-035",
        "disease": "Influenza",
        "district": "ALL",
        "start_date": "2024-11-01",
        "end_date": "2025-01-15",
        "severity": "Critical",
        "outbreak_type": "nationwide",
    })

    pd.DataFrame(events).to_csv(
        OUTBREAK_FILE,
        index=False
    )

    print(f"Outbreak events created: {len(events)}")

    return events


def random_date(start, end):
    days = (end - start).days
    return start + timedelta(days=random.randint(0, days))


# ============================================================
# FACILITY REPORTING SILENCE
# ============================================================

def generate_silence_events(facilities):
    print("Generating facility reporting-silence events...")

    all_facilities = [
        facility
        for facility_list in facilities.values()
        for facility in facility_list
    ]

    selected = random.sample(
        all_facilities,
        min(19, len(all_facilities))
    )

    events = []

    for facility in selected:
        start = random_date(
            START_DATE + timedelta(days=30),
            END_DATE - timedelta(days=30)
        )

        end = start + timedelta(days=20)

        events.append({
            "facility_id": facility["facility_id"],
            "district": facility["district"],
            "silence_start": start.isoformat(),
            "silence_end": end.isoformat(),
        })

    pd.DataFrame(events).to_csv(
        SILENCE_FILE,
        index=False
    )

    print(f"Facility silence events created: {len(events)}")

    return events


# ============================================================
# OUTBREAK MULTIPLIER
# ============================================================

def outbreak_multiplier(district, day, events):
    multiplier = 1.0

    for event in events:
        if event["district"] not in ("ALL", district):
            continue

        start = date.fromisoformat(event["start_date"])
        end = date.fromisoformat(event["end_date"])

        if not start <= day <= end:
            continue

        duration = max(1, (end - start).days)
        progress = (day - start).days / duration

        peak = (
            10.0
            if event["outbreak_type"] == "nationwide"
            else SEVERITY_PEAK[event["severity"]]
        )

        if event["outbreak_type"] == "slow_growth":
            curve = progress ** 1.5
        else:
            curve = np.sin(np.pi * progress)

        multiplier = max(
            multiplier,
            1.0 + (peak - 1.0) * max(0.0, curve)
        )

    return min(multiplier, 25.0)


# ============================================================
# WEATHER EFFECTS
# ============================================================

def climate_effects(code, day, weather):
    dengue_rain = []
    dengue_temp = []
    recent_rain = []

    # Dengue: rainfall and temperature 28-56 days earlier.
    for lag in range(28, 57, 7):
        old_day = day - timedelta(days=lag)

        if (code, old_day) in weather:
            rain, temp = weather[(code, old_day)]
            dengue_rain.append(rain)
            dengue_temp.append(temp)

    # Diarrhoea: recent rainfall.
    for lag in range(1, 8):
        old_day = day - timedelta(days=lag)

        if (code, old_day) in weather:
            rain, _ = weather[(code, old_day)]
            recent_rain.append(rain)

    rain_lag = (
        np.mean(dengue_rain) if dengue_rain else 0.0
    )

    temp_lag = (
        np.mean(dengue_temp) if dengue_temp else 27.0
    )

    rain_recent = sum(recent_rain)

    dengue_effect = (
        1.0
        + min(rain_lag / 25.0, 1.5)
        + max(0.0, temp_lag - 27.0) * 0.04
    )

    diarrhoea_effect = (
        1.0 + min(rain_recent / 100.0, 2.0)
    )

    return dengue_effect, diarrhoea_effect


# ============================================================
# SEASONAL AND DISEASE WEIGHTS
# ============================================================

def disease_weights(code, day, events, weather):
    month = day.month

    dengue = 1.0
    diarrhoea = 1.0
    influenza = 1.0
    malaria = 0.7
    other = 1.2

    # Dengue season.
    if month in [6, 7, 8, 9, 10]:
        dengue *= 2.8

    # Diarrhoea season.
    if month in [4, 5, 6, 7, 8, 9]:
        diarrhoea *= 1.7

    # Influenza cooler season.
    if month in [11, 12, 1, 2]:
        influenza *= 1.8

    # Additional malaria weight in selected hill districts.
    if code in ["BD2003", "BD2004", "BD2006", "BD2007"]:
        if month in [5, 6, 7, 8, 9]:
            malaria *= 3.0

    dengue_weather, diarrhoea_weather = climate_effects(
        code,
        day,
        weather
    )

    dengue *= dengue_weather
    diarrhoea *= diarrhoea_weather

    weights = {
        "Dengue": dengue,
        "Diarrhoea": diarrhoea,
        "Influenza": influenza,
        "Malaria": malaria,
        "Other": other,
    }

    # Disease-specific outbreak effects.
    for event in events:
        if event["district"] not in ("ALL", code):
            continue

        start = date.fromisoformat(event["start_date"])
        end = date.fromisoformat(event["end_date"])

        if start <= day <= end:
            duration = max(1, (end - start).days)
            progress = (day - start).days / duration

            peak = (
                10.0
                if event["outbreak_type"] == "nationwide"
                else SEVERITY_PEAK[event["severity"]]
            )

            if event["outbreak_type"] == "slow_growth":
                curve = progress ** 1.5
            else:
                curve = np.sin(np.pi * progress)

            weights[event["disease"]] *= (
                1.0 + (peak - 1.0) * max(0.0, curve)
            )

    return np.array(
        [weights[disease] for disease in DISEASES],
        dtype=float
    )


# ============================================================
# NEGATIVE BINOMIAL DISTRICT-DAY COUNTS
# ============================================================

def generate_daily_counts(
    districts,
    district_population,
    events,
    weather
):
    print("Building district-day incidence model...")

    days = date_range(START_DATE, END_DATE)

    total_population = sum(
        district_population.get(code, 0.0)
        for code in districts
    )

    if total_population <= 0:
        raise ValueError("District population totals are invalid.")

    cells = []

    for code in districts:
        population = district_population.get(code, 0.0)

        if population <= 0:
            raise ValueError(
                f"Missing population for district code {code}"
            )

        for day in days:
            cells.append((code, day, population))

    print(f"District-day combinations: {len(cells)}")

    baseline_daily_total = TEST_RECORDS / len(days)

    mu = np.zeros(len(cells), dtype=float)
    weights = np.zeros(
        (len(cells), len(DISEASES)),
        dtype=float
    )

    for i, (code, day, population) in enumerate(cells):
        population_share = population / total_population

        outbreak_factor = outbreak_multiplier(
            code,
            day,
            events
        )

        mu[i] = (
            baseline_daily_total
            * population_share
            * outbreak_factor
        )

        weights[i] = disease_weights(
            code,
            day,
            events,
            weather
        )

    # Gamma-Poisson mixture produces a Negative Binomial count.
    dispersion_k = 2.0

    gamma_rates = rng.gamma(
        shape=dispersion_k,
        scale=np.maximum(mu, 1e-10) / dispersion_k
    )

    raw_counts = rng.poisson(gamma_rates)

    if raw_counts.sum() == 0:
        raise RuntimeError(
            "All generated district-day counts are zero."
        )

    # Scale the NB allocation to the requested exact test size.
    scaled = raw_counts * (
        TEST_RECORDS / raw_counts.sum()
    )

    counts = np.floor(scaled).astype(np.int64)

    remainder = TEST_RECORDS - int(counts.sum())

    if remainder > 0:
        fractions = scaled - counts
        chosen = np.argsort(fractions)[-remainder:]
        counts[chosen] += 1

    print(f"Base encounters allocated: {counts.sum():,}")
    print(
        f"District-day cells with cases: "
        f"{(counts > 0).sum():,}"
    )

    positive = counts[counts > 0]

    if len(positive):
        print(
            f"Mean positive district-day count: "
            f"{positive.mean():.2f}"
        )
        print(
            f"Variance positive district-day count: "
            f"{positive.var():.2f}"
        )

    return cells, counts, weights


# ============================================================
# BUILD ENCOUNTER RECORDS
# ============================================================

def build_encounters(
    cells,
    counts,
    weights,
    facilities,
    age_profiles,
    silence_events
):
    print("Generating encounter records...")

    records = []

    silence_lookup = {}

    for event in silence_events:
        silence_lookup[event["facility_id"]] = (
            date.fromisoformat(event["silence_start"]),
            date.fromisoformat(event["silence_end"])
        )

    # Cache profile weights to avoid recalculating them per case.
    profile_cache = {}

    for code, profile in age_profiles.items():
        profile_cache[code] = np.array(
            [item[3] for item in profile],
            dtype=float
        )

        profile_cache[code] /= profile_cache[code].sum()

    for cell_index, count in enumerate(counts):
        if count == 0:
            continue

        code, onset, population = cells[cell_index]

        district_facilities = facilities.get(code, [])
        profile = age_profiles.get(code, [])

        if not district_facilities:
            raise ValueError(
                f"No facilities for district code {code}"
            )

        if not profile:
            raise ValueError(
                f"No age-sex profile for district code {code}"
            )

        disease_prob = weights[cell_index]
        disease_prob = disease_prob / disease_prob.sum()

        disease_indices = rng.choice(
            len(DISEASES),
            size=int(count),
            p=disease_prob
        )

        selected_profiles = rng.choice(
            len(profile),
            size=int(count),
            p=profile_cache[code]
        )

        for disease_index, profile_index in zip(
            disease_indices,
            selected_profiles
        ):
            disease = DISEASES[disease_index]

            facility = random.choice(district_facilities)
            facility_id = facility["facility_id"]

            low, high, sex, _ = profile[profile_index]
            age = random.randint(low, high)

            # Right-skewed reporting delay.
            delay = min(
                int(rng.exponential(scale=3.0)),
                21
            )

            report_day = onset + timedelta(days=delay)

            # Friday and Saturday reporting delays.
            if report_day.weekday() in [4, 5]:
                report_day += timedelta(
                    days=random.randint(1, 2)
                )

            # Occasional holiday-like delay.
            if random.random() < 0.02:
                report_day += timedelta(
                    days=random.randint(1, 3)
                )

            # Facility silence and backfill.
            silence = silence_lookup.get(facility_id)

            if silence:
                silence_start, silence_end = silence

                if silence_start <= onset <= silence_end:
                    report_day = max(
                        report_day,
                        silence_end + timedelta(days=1)
                    )

                elif silence_start <= report_day <= silence_end:
                    report_day = (
                        silence_end + timedelta(days=1)
                    )

            # Maximum reporting delay: 21 days.
            report_day = min(
                report_day,
                onset + timedelta(days=21)
            )

            icd10 = random.choice(
                DISEASE_CODES[disease]
            )

            lab_confirmed = (
                "True" if random.random() < 0.35
                else "False"
            )

            admitted = (
                "True" if random.random() < 0.18
                else "False"
            )

            outcome = (
                "Deceased" if random.random() < 0.005
                else "Recovered"
            )

            records.append({
                "encounter_id": (
                    f"ENC-{len(records) + 1:010d}"
                ),
                "patient_id": (
                    f"PAT-{random.randint(100000, 999999)}"
                ),
                "facility_id": facility_id,
                "district": facility["district"],
                "onset_date": onset.isoformat(),
                "report_date": report_day.isoformat(),
                "age_years": age,
                "sex": sex,
                "icd10_code": icd10,
                "lab_confirmed": lab_confirmed,
                "admitted": admitted,
                "outcome": outcome,
            })

    df = pd.DataFrame(
        records,
        columns=ENCOUNTER_COLUMNS
    )

    # Intentional quality defects.
    n = len(df)

    invalid_age_count = max(1, int(n * 0.01))
    invalid_sex_count = max(1, int(n * 0.01))
    orphan_count = max(1, int(n * 0.005))

    age_indices = rng.choice(
        n,
        size=invalid_age_count,
        replace=False
    )

    df.loc[age_indices, "age_years"] = 212

    sex_indices = rng.choice(
        n,
        size=invalid_sex_count,
        replace=False
    )

    df.loc[
        sex_indices,
        "sex"
    ] = "Unknown_Invalid"

    orphan_indices = rng.choice(
        n,
        size=orphan_count,
        replace=False
    )

    df.loc[
        orphan_indices,
        "facility_id"
    ] = "FAC-ORPHAN-999999"

    # Approximately 2% duplicate/resubmitted records.
    duplicate_count = max(1, int(n * 0.02))

    duplicate_indices = rng.choice(
        n,
        size=duplicate_count,
        replace=False
    )

    duplicates = df.iloc[duplicate_indices].copy()

    df = pd.concat(
        [df, duplicates],
        ignore_index=True
    )

    return df


# ============================================================
# MAIN
# ============================================================

def main():
    ensure_directories()

    print("=" * 65)
    print("EPISCOPE - IMPROVED ENCOUNTER GENERATOR")
    print("=" * 65)

    print(f"Target base encounters: {TEST_RECORDS:,}")
    print(f"Onset period: {START_DATE} to {END_DATE}")
    print("Statistical model: Gamma-Poisson / Negative Binomial")
    print()

    facilities = load_facilities()

    age_profiles, district_population = load_population()

    weather = load_weather()

    districts = sorted(
        set(facilities.keys())
        & set(age_profiles.keys())
        & set(district_population.keys())
    )

    print(f"Districts used: {len(districts)}")

    if len(districts) != 64:
        raise ValueError(
            f"Expected 64 districts but found {len(districts)}. "
            "Check district codes across source files."
        )

    events = generate_outbreaks(districts)

    silence_events = generate_silence_events(facilities)

    cells, counts, weights = generate_daily_counts(
        districts,
        district_population,
        events,
        weather
    )

    df = build_encounters(
        cells,
        counts,
        weights,
        facilities,
        age_profiles,
        silence_events
    )

    print()
    print("Writing encounter CSV...")

    df.to_csv(
        ENCOUNTER_FILE,
        index=False,
        columns=ENCOUNTER_COLUMNS
    )

    print()
    print("=" * 65)
    print("ENCOUNTER GENERATION COMPLETE")
    print("=" * 65)

    print(f"Base encounters: {TEST_RECORDS:,}")
    print(f"Final rows including duplicates: {len(df):,}")
    print(
        f"Duplicate rows added: "
        f"{len(df) - TEST_RECORDS:,}"
    )

    print(f"Output: {ENCOUNTER_FILE}")
    print(f"Outbreak ground truth: {OUTBREAK_FILE}")
    print(f"Facility silence events: {SILENCE_FILE}")

    print()
    print("Disease distribution:")
    print(df["icd10_code"].value_counts().to_string())

    print()
    print("District distribution:")
    print(
        df["district"]
        .value_counts()
        .head(10)
        .to_string()
    )

    print()
    print("Reporting delay summary:")

    delays = (
        pd.to_datetime(df["report_date"])
        - pd.to_datetime(df["onset_date"])
    ).dt.days

    print(delays.describe().to_string())

    print()
    print("NEXT: Run generator\\verify_encounters.py")


if __name__ == "__main__":
    main()
