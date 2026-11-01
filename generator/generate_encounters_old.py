
import csv
import random
import uuid
from pathlib import Path
from datetime import date, timedelta
from collections import Counter, defaultdict
from bisect import bisect_right

import pandas as pd


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FACILITY_FILE = (
    PROJECT_ROOT / "data" / "bronze" /
    "facilities" / "facilities.csv"
)

POPULATION_FILE = (
    PROJECT_ROOT / "data" / "reference" /
    "district_population.csv"
)

WEATHER_FILE = (
    PROJECT_ROOT / "data" / "bronze" /
    "weather" / "weather_daily.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "bronze" / "encounters"
)

OUTPUT_FILE = OUTPUT_DIR / "encounters_test.csv"

GROUND_TRUTH_DIR = (
    PROJECT_ROOT / "data" / "ground_truth"
)

OUTBREAK_FILE = (
    GROUND_TRUTH_DIR / "ground_truth_outbreaks.csv"
)

SILENCE_FILE = (
    GROUND_TRUTH_DIR / "facility_silence_events.csv"
)

SEED = 42

# Keep this small for testing.
# We will increase the scale after validation.
TEST_RECORDS = 10000

START_DATE = date(2021, 1, 1)
END_DATE = date(2025, 12, 31)

rng = random.Random(SEED)


# ============================================================
# DISEASE CONFIGURATION
# ============================================================

DISEASE_CODES = {
    "Dengue": ["A90", "A91"],
    "Diarrhoea": ["A00", "A09"],
    "Influenza": ["J10", "J11"],
    "Malaria": ["B50", "B51", "B52", "B53", "B54"],
    "Other": ["Z00", "R50", "B99"]
}

BASE_DISEASE_WEIGHTS = {
    "Dengue": 0.20,
    "Diarrhoea": 0.20,
    "Influenza": 0.20,
    "Malaria": 0.10,
    "Other": 0.30
}

HILL_DISTRICTS = {
    "Bandarban",
    "Rangamati",
    "Khagrachhari"
}

HIGH_DENGUE_AMPLITUDE_DISTRICTS = {
    "Dhaka",
    "Chattogram",
    "Gazipur",
    "Narayanganj",
    "Khulna",
    "Rajshahi",
    "Sylhet"
}

COLUMNS = [
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
    "outcome"
]

OUTBREAK_COLUMNS = [
    "outbreak_id",
    "disease",
    "district",
    "start_date",
    "end_date",
    "severity",
    "outbreak_type"
]


# ============================================================
# LOAD FACILITIES
# ============================================================

def load_facilities():

    with open(FACILITY_FILE, encoding="utf-8-sig") as file:
        facilities = list(csv.DictReader(file))

    if not facilities:
        raise ValueError("No facilities found.")

    return facilities


# ============================================================
# LOAD WORLDPOP AGE-SEX DISTRIBUTION
# ============================================================

def load_population():

    df = pd.read_csv(POPULATION_FILE)

    required = {
        "district_code",
        "age_class",
        "male_population",
        "female_population"
    }

    if not required.issubset(df.columns):
        raise ValueError(
            "WorldPop file is missing required columns."
        )

    population = defaultdict(list)

    for _, row in df.iterrows():

        age_class = int(row["age_class"])

        # Avoid overlapping 85+ and 90+ categories.
        if age_class > 80:
            continue

        lower_age = age_class

        if age_class == 0:
            upper_age = 1

        elif age_class == 80:
            upper_age = 96

        else:
            upper_age = age_class + 5

        district_code = str(row["district_code"])

        male = max(
            0,
            float(row["male_population"])
        )

        female = max(
            0,
            float(row["female_population"])
        )

        if male > 0:
            population[district_code].append(
                (lower_age, upper_age, "Male", male)
            )

        if female > 0:
            population[district_code].append(
                (lower_age, upper_age, "Female", female)
            )

    if not population:
        raise ValueError(
            "No usable WorldPop population records found."
        )

    return population


def sample_demographics(district_code, population):

    groups = population.get(str(district_code))

    if not groups:
        return (
            rng.randint(0, 95),
            rng.choice(["Male", "Female"])
        )

    selected = rng.choices(
        groups,
        weights=[group[3] for group in groups],
        k=1
    )[0]

    lower_age, upper_age, sex, _ = selected

    age = rng.randint(
        lower_age,
        upper_age - 1
    )

    return age, sex


# ============================================================
# LOAD HISTORICAL WEATHER
# ============================================================

def load_weather():

    df = pd.read_csv(WEATHER_FILE)

    required = {
        "district_code",
        "date",
        "precipitation_mm",
        "temperature_mean_c",
        "humidity_mean_percent"
    }

    if not required.issubset(df.columns):
        raise ValueError(
            "Weather file is missing required columns."
        )

    weather = defaultdict(dict)

    for _, row in df.iterrows():

        district_code = str(row["district_code"])

        weather_date = date.fromisoformat(
            str(row["date"])
        )

        weather[district_code][weather_date] = {
            "rain": float(row["precipitation_mm"]),
            "temperature": float(
                row["temperature_mean_c"]
            ),
            "humidity": float(
                row["humidity_mean_percent"]
            )
        }

    return weather


def get_climate_effect(district_code, onset, weather):

    district_weather = weather.get(
        str(district_code),
        {}
    )

    # Dengue: rainfall and temperature
    # 28-56 days before onset.

    lagged_rain = []
    lagged_temperature = []

    for days_ago in range(28, 57):

        historical_date = (
            onset - timedelta(days=days_ago)
        )

        values = district_weather.get(
            historical_date
        )

        if values:
            lagged_rain.append(values["rain"])
            lagged_temperature.append(
                values["temperature"]
            )

    # Diarrhoea: rainfall during previous 7 days.

    recent_rain = []

    for days_ago in range(0, 7):

        historical_date = (
            onset - timedelta(days=days_ago)
        )

        values = district_weather.get(
            historical_date
        )

        if values:
            recent_rain.append(values["rain"])

    dengue_multiplier = 1.0
    diarrhoea_multiplier = 1.0

    if lagged_rain:

        average_lagged_rain = (
            sum(lagged_rain) / len(lagged_rain)
        )

        average_lagged_temperature = (
            sum(lagged_temperature)
            / len(lagged_temperature)
        )

        dengue_multiplier *= (
            1.0 + min(
                1.5,
                average_lagged_rain / 8.0
            )
        )

        dengue_multiplier *= max(
            0.7,
            min(
                1.3,
                1.0 + (
                    average_lagged_temperature - 26.0
                ) * 0.025
            )
        )

    if recent_rain:

        recent_rain_total = sum(recent_rain)

        diarrhoea_multiplier *= (
            1.0 + min(
                2.0,
                recent_rain_total / 100.0
            )
        )

    return dengue_multiplier, diarrhoea_multiplier


# ============================================================
# GENERATE SIMULATED OUTBREAKS
# ============================================================

def generate_outbreaks(facilities):

    districts = sorted({
        facility["district"]
        for facility in facilities
    })

    outbreaks = []

    # 34 local outbreaks.

    for index in range(34):

        district = rng.choice(districts)

        if district in HILL_DISTRICTS:

            disease = rng.choices(
                [
                    "Malaria",
                    "Dengue",
                    "Diarrhoea",
                    "Influenza"
                ],
                weights=[
                    0.55,
                    0.15,
                    0.15,
                    0.15
                ],
                k=1
            )[0]

        else:

            disease = rng.choice([
                "Dengue",
                "Diarrhoea",
                "Influenza",
                "Malaria"
            ])

        duration = rng.randint(10, 45)

        start_offset = rng.randint(
            0,
            (END_DATE - START_DATE).days - duration
        )

        start = (
            START_DATE + timedelta(days=start_offset)
        )

        # Duration is inclusive.
        end = start + timedelta(days=duration - 1)

        outbreak_type = (
            "slow_growth"
            if index == 0
            else "regional"
        )

        outbreaks.append({
            "outbreak_id": f"OUTBREAK_{index + 1:03d}",
            "disease": disease,
            "district": district,
            "start_date": start,
            "end_date": end,
            "severity": rng.choice([
                "Moderate",
                "High",
                "Critical"
            ]),
            "outbreak_type": outbreak_type
        })

    # One nationwide simulated novel respiratory outbreak.

    national_duration = 45

    national_offset = rng.randint(
        0,
        (END_DATE - START_DATE).days - national_duration
    )

    national_start = (
        START_DATE + timedelta(days=national_offset)
    )

    outbreaks.append({
        "outbreak_id": "OUTBREAK_035",
        "disease": "Influenza",
        "district": "NATIONAL",
        "start_date": national_start,
        "end_date": national_start + timedelta(
            days=national_duration - 1
        ),
        "severity": "Critical",
        "outbreak_type": "nationwide_novel_respiratory"
    })

    return outbreaks


def save_outbreaks(outbreaks):

    GROUND_TRUTH_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTBREAK_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=OUTBREAK_COLUMNS
        )

        writer.writeheader()

        for outbreak in outbreaks:

            row = outbreak.copy()

            row["start_date"] = (
                row["start_date"].isoformat()
            )

            row["end_date"] = (
                row["end_date"].isoformat()
            )

            writer.writerow(row)


# ============================================================
# OUTBREAK INTENSITY MODEL
# ============================================================

def outbreak_curve(outbreak, current_date):

    start = outbreak["start_date"]
    end = outbreak["end_date"]

    if not start <= current_date <= end:
        return 0.0

    duration = max(
        1,
        (end - start).days
    )

    progress = (
        (current_date - start).days / duration
    )

    if outbreak["outbreak_type"] == "slow_growth":

        # Gradual increase in outbreak intensity.
        return progress ** 1.5

    # Regional and nationwide outbreaks follow
    # a rise-and-fall epidemic curve.
    import math

    return max(
        0.0,
        math.sin(math.pi * progress)
    )


def outbreak_peak_multiplier(outbreak):

    severity_multipliers = {
        "Moderate": 4.0,
        "High": 6.0,
        "Critical": 8.0
    }

    if (
        outbreak["outbreak_type"]
        == "nationwide_novel_respiratory"
    ):
        return 10.0

    return severity_multipliers.get(
        outbreak["severity"],
        4.0
    )


def outbreak_rate_multiplier(
    district,
    current_date,
    outbreaks_by_district,
    national_outbreaks
):

    relevant_outbreaks = (
        outbreaks_by_district.get(district, [])
        + national_outbreaks
    )

    multiplier = 1.0

    for outbreak in relevant_outbreaks:

        curve = outbreak_curve(
            outbreak,
            current_date
        )

        if curve <= 0:
            continue

        peak = outbreak_peak_multiplier(
            outbreak
        )

        event_multiplier = (
            1.0 + (peak - 1.0) * curve
        )

        multiplier *= event_multiplier

    # Prevent unrealistic multiplication when events overlap.
    return min(multiplier, 25.0)


def index_outbreaks(outbreaks):

    outbreaks_by_district = defaultdict(list)
    national_outbreaks = []

    for outbreak in outbreaks:

        if outbreak["district"] == "NATIONAL":

            national_outbreaks.append(outbreak)

        else:

            outbreaks_by_district[
                outbreak["district"]
            ].append(outbreak)

    return outbreaks_by_district, national_outbreaks


# ============================================================
# BUILD WEIGHTED DISTRICT-DATE SAMPLING INDEX
# ============================================================

def build_sampling_index(
    facilities,
    population,
    outbreaks
):

    print("Building outbreak-aware sampling index...")

    outbreaks_by_district, national_outbreaks = (
        index_outbreaks(outbreaks)
    )

    district_lookup = {}

    for facility in facilities:

        code = str(facility["district_code"])

        district_lookup[code] = (
            facility["district"]
        )

    district_population = {}

    for code, groups in population.items():

        district_population[code] = sum(
            group[3] for group in groups
        )

    sampling_entries = []
    cumulative_weights = []

    cumulative_total = 0.0

    current_date = START_DATE

    while current_date <= END_DATE:

        for district_code, district in district_lookup.items():

            # Population-weighted baseline.
            base_weight = district_population.get(
                district_code,
                1.0
            )

            if base_weight <= 0:
                base_weight = 1.0

            # Increase sampling probability during outbreaks.
            outbreak_multiplier = (
                outbreak_rate_multiplier(
                    district,
                    current_date,
                    outbreaks_by_district,
                    national_outbreaks
                )
            )

            final_weight = (
                base_weight * outbreak_multiplier
            )

            cumulative_total += final_weight

            sampling_entries.append((
                district_code,
                district,
                current_date
            ))

            cumulative_weights.append(
                cumulative_total
            )

        current_date += timedelta(days=1)

    if not sampling_entries:
        raise ValueError(
            "Could not build the district-date sampling index."
        )

    print(
        "Sampling combinations:",
        len(sampling_entries)
    )

    return (
        sampling_entries,
        cumulative_weights,
        cumulative_total,
        outbreaks_by_district,
        national_outbreaks
    )


def sample_district_date(
    sampling_entries,
    cumulative_weights,
    cumulative_total
):

    random_weight = rng.random() * cumulative_total

    index = bisect_right(
        cumulative_weights,
        random_weight
    )

    if index >= len(sampling_entries):
        index = len(sampling_entries) - 1

    return sampling_entries[index]


# ============================================================
# SIMULATE FACILITY REPORTING SILENCE
# ============================================================

def generate_facility_silence(facilities):

    silence_events = []

    # Approximately 3% of facilities experience silence.

    selected_count = max(
        1,
        round(len(facilities) * 0.03)
    )

    selected_facilities = rng.sample(
        facilities,
        selected_count
    )

    for facility in selected_facilities:

        start_offset = rng.randint(
            0,
            (END_DATE - START_DATE).days - 21
        )

        silence_start = (
            START_DATE + timedelta(days=start_offset)
        )

        silence_end = (
            silence_start + timedelta(days=20)
        )

        silence_events.append({
            "facility_id": facility["facility_id"],
            "district": facility["district"],
            "silence_start": silence_start,
            "silence_end": silence_end
        })

    return silence_events


def save_silence_events(events):

    GROUND_TRUTH_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    columns = [
        "facility_id",
        "district",
        "silence_start",
        "silence_end"
    ]

    with open(
        SILENCE_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=columns
        )

        writer.writeheader()

        for event in events:

            row = event.copy()

            row["silence_start"] = (
                row["silence_start"].isoformat()
            )

            row["silence_end"] = (
                row["silence_end"].isoformat()
            )

            writer.writerow(row)


# ============================================================
# REPORTING DELAYS
# ============================================================

def reporting_delay(onset, facility_id, silence_lookup):

    # Ordinary reporting delay: right-skewed, 0-21 days.
    delay = min(
        int(rng.expovariate(1 / 3)),
        21
    )

    report = onset + timedelta(days=delay)

    # Bangladesh weekend: Friday and Saturday.
    while report.weekday() in (4, 5):
        report += timedelta(days=1)

    # Occasional simulated holiday delay.
    if rng.random() < 0.02:
        report += timedelta(
            days=rng.randint(1, 3)
        )

    # Facility silence followed by backfill.
    silence = silence_lookup.get(facility_id)

    if silence:

        silence_start, silence_end = silence

        if silence_start <= onset <= silence_end:

            backfill_date = (
                silence_end + timedelta(
                    days=rng.randint(0, 2)
                )
            )

            report = max(
                report,
                backfill_date
            )

    # Enforce the absolute 21-day reporting limit.
    maximum_report_date = (
        onset + timedelta(days=21)
    )

    report = min(
        report,
        maximum_report_date
    )

    # Keep report date within the simulation period.
    report = min(
        report,
        END_DATE
    )

    return report


# ============================================================
# DISEASE SELECTION
# ============================================================

def choose_disease(
    onset,
    district,
    district_code,
    weather,
    outbreaks_by_district,
    national_outbreaks
):

    weights = BASE_DISEASE_WEIGHTS.copy()

    month = onset.month

    # Annual disease seasonality.

    if month in [6, 7, 8, 9, 10]:
        weights["Dengue"] *= 2.5

    if month in [4, 5, 6, 7, 8, 9]:
        weights["Diarrhoea"] *= 1.5

    if month in [12, 1, 2]:
        weights["Influenza"] *= 1.5

    if district in HILL_DISTRICTS:
        weights["Malaria"] *= 3.0

    if district in HIGH_DENGUE_AMPLITUDE_DISTRICTS:
        weights["Dengue"] *= 1.5

    # Historical weather effects.

    dengue_effect, diarrhoea_effect = (
        get_climate_effect(
            district_code,
            onset,
            weather
        )
    )

    weights["Dengue"] *= dengue_effect
    weights["Diarrhoea"] *= diarrhoea_effect

    # Disease-specific outbreak effects.

    relevant_outbreaks = (
        outbreaks_by_district.get(district, [])
        + national_outbreaks
    )

    for outbreak in relevant_outbreaks:

        if outbreak["disease"] not in weights:
            continue

        curve = outbreak_curve(
            outbreak,
            onset
        )

        if curve <= 0:
            continue

        peak = outbreak_peak_multiplier(
            outbreak
        )

        disease_multiplier = (
            1.0 + (peak - 1.0) * curve
        )

        weights[outbreak["disease"]] *= (
            disease_multiplier
        )

    # Gamma-distributed variation for extra heterogeneity.

    for disease in weights:

        gamma_multiplier = rng.gammavariate(
            4.0,
            0.25
        )

        weights[disease] *= gamma_multiplier

    diseases = list(weights.keys())

    return rng.choices(
        diseases,
        weights=[
            weights[disease]
            for disease in diseases
        ],
        k=1
    )[0]


# ============================================================
# GENERATE ENCOUNTERS
# ============================================================

def generate_encounters():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    GROUND_TRUTH_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Loading facilities...")
    facilities = load_facilities()

    print("Loading WorldPop age-sex population...")
    population = load_population()

    print("Loading historical weather...")
    weather = load_weather()

    print("Generating outbreak ground truth...")
    outbreaks = generate_outbreaks(facilities)
    save_outbreaks(outbreaks)

    print("Generating facility silence events...")
    silence_events = generate_facility_silence(
        facilities
    )

    save_silence_events(silence_events)

    silence_lookup = {
        event["facility_id"]: (
            event["silence_start"],
            event["silence_end"]
        )
        for event in silence_events
    }

    # Group facilities by district code.

    facilities_by_district = defaultdict(list)

    for facility in facilities:

        facilities_by_district[
            str(facility["district_code"])
        ].append(facility)

    # Build the outbreak-aware sampling index.

    (
        sampling_entries,
        cumulative_weights,
        cumulative_total,
        outbreaks_by_district,
        national_outbreaks
    ) = build_sampling_index(
        facilities,
        population,
        outbreaks
    )

    print("\nGenerating encounters...")

    duplicate_count = 0
    invalid_age_count = 0
    invalid_sex_count = 0
    orphan_count = 0

    disease_counts = Counter()
    district_counts = Counter()
    outbreak_period_counts = Counter()

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=COLUMNS
        )

        writer.writeheader()

        for i in range(TEST_RECORDS):

            # Select district and date using population
            # and outbreak-aware weights.

            district_code, district, onset = (
                sample_district_date(
                    sampling_entries,
                    cumulative_weights,
                    cumulative_total
                )
            )

            # Select a facility within that district.

            facility = rng.choice(
                facilities_by_district[district_code]
            )

            disease = choose_disease(
                onset,
                district,
                district_code,
                weather,
                outbreaks_by_district,
                national_outbreaks
            )

            report = reporting_delay(
                onset,
                facility["facility_id"],
                silence_lookup
            )

            age, sex = sample_demographics(
                district_code,
                population
            )

            code = rng.choice(
                DISEASE_CODES[disease]
            )

            lab_probability = (
                0.35
                if disease != "Other"
                else 0.05
            )

            lab_confirmed = (
                rng.random() < lab_probability
            )

            admission_probability = (
                0.30
                if disease in ["Dengue", "Malaria"]
                else 0.15
            )

            admitted = (
                rng.random() < admission_probability
            )

            if rng.random() < 0.005:
                outcome = "Deceased"

            elif admitted:
                outcome = "Admitted"

            else:
                outcome = "Recovered"

            facility_id = facility["facility_id"]

            # Approximately 0.5% orphan facilities.

            if rng.random() < 0.005:

                facility_id = (
                    "UNKNOWN_" + str(i)
                )

                orphan_count += 1

            # Approximately 1% invalid ages.

            if rng.random() < 0.01:

                age = 212

                invalid_age_count += 1

            # Approximately 1% invalid sex values.

            if rng.random() < 0.01:

                sex = "Unknown_Invalid"

                invalid_sex_count += 1

            record = {
                "encounter_id": str(
                    uuid.UUID(
                        int=rng.getrandbits(128)
                    )
                ),
                "patient_id": (
                    f"SYNTH{rng.randint(100000, 999999)}"
                ),
                "facility_id": facility_id,
                "district": district,
                "onset_date": onset.isoformat(),
                "report_date": report.isoformat(),
                "age_years": age,
                "sex": sex,
                "icd10_code": code,
                "lab_confirmed": lab_confirmed,
                "admitted": admitted,
                "outcome": outcome
            }

            writer.writerow(record)

            disease_counts[disease] += 1
            district_counts[district] += 1

            # Count encounters that fall inside
            # a known outbreak period.
            # This is only a generation diagnostic.

            for outbreak in (
                outbreaks_by_district.get(district, [])
                + national_outbreaks
            ):

                if (
                    outbreak["start_date"]
                    <= onset
                    <= outbreak["end_date"]
                ):

                    outbreak_period_counts[
                        outbreak["outbreak_id"]
                    ] += 1

            # Approximately 2% duplicate submissions.

            if rng.random() < 0.02:

                writer.writerow(record.copy())

                duplicate_count += 1

            if (i + 1) % 2000 == 0:

                print(
                    f"Generated {i + 1:,} / "
                    f"{TEST_RECORDS:,} base records"
                )

    # ========================================================
    # GENERATION SUMMARY
    # ========================================================

    print("\n" + "=" * 55)
    print("EPISCOPE - ENCOUNTER GENERATION COMPLETE")
    print("=" * 55)

    print("Base records:", TEST_RECORDS)

    print(
        "Total rows including duplicates:",
        TEST_RECORDS + duplicate_count
    )

    print("Duplicate submissions:", duplicate_count)
    print("Invalid ages:", invalid_age_count)
    print("Invalid sex values:", invalid_sex_count)
    print("Orphan facility IDs:", orphan_count)

    print("\nDisease distribution:")

    for disease, count in sorted(
        disease_counts.items()
    ):
        print(f"{disease}: {count}")

    print("\nTop 10 districts by encounter count:")

    for district, count in district_counts.most_common(10):
        print(f"{district}: {count}")

    print("\nEncounters during outbreak periods:")

    for outbreak_id, count in sorted(
        outbreak_period_counts.items()
    ):
        print(f"{outbreak_id}: {count}")

    print("\nOutbreak events:", len(outbreaks))

    print(
        "Facility silence events:",
        len(silence_events)
    )

    print("\nEncounter output:", OUTPUT_FILE)
    print("Outbreak ground truth:", OUTBREAK_FILE)
    print("Facility silence ground truth:", SILENCE_FILE)

    print("\nGeneration finished successfully.")


if __name__ == "__main__":
    generate_encounters()
