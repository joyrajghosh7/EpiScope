
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

SCORES_DIR = (
    ROOT / "data" / "gold" /
    "outbreak_detection_improved" / "weekly_scores"
)

GROUND_TRUTH = (
    ROOT / "data" / "ground_truth" /
    "ground_truth_outbreaks.csv"
)

OUTPUT_DIR = ROOT / "data" / "gold" / "threshold_tuning"

THRESHOLDS = [2.0, 2.5, 3.0]
MIN_CASES = 5
MIN_SEASONAL_YEARS = 3
TOLERANCE_DAYS = 7


def create_episodes(alerts):

    if alerts.empty:
        return pd.DataFrame()

    alerts = alerts.copy()

    alerts["week_start"] = pd.to_datetime(alerts["week_start"])

    alerts = alerts.sort_values(
        ["district_code", "disease", "week_start"]
    )

    groups = ["district_code", "disease"]

    previous = alerts.groupby(groups)["week_start"].shift(1)

    new_episode = (
        previous.isna()
        | ((alerts["week_start"] - previous).dt.days > 7)
    )

    alerts["_episode"] = (
        new_episode.astype(int)
        .groupby([
            alerts["district_code"],
            alerts["disease"]
        ]).cumsum()
    )

    episodes = (
        alerts.groupby(
            ["district_code", "district", "disease", "_episode"]
        )
        .agg(
            episode_start=("week_start", "min"),
            episode_end=("week_start", "max"),
            alert_weeks=("week_start", "count"),
            peak_cases=("total_cases", "max")
        )
        .reset_index()
        .drop(columns="_episode")
    )

    episodes["episode_end"] += pd.Timedelta(days=6)

    return episodes


def evaluate(episodes, events):

    matched_indices = set()
    event_detected = []

    for _, event in events.iterrows():

        start = event["start_date"] - pd.Timedelta(
            days=TOLERANCE_DAYS
        )

        end = event["end_date"] + pd.Timedelta(
            days=TOLERANCE_DAYS
        )

        disease_match = (
            episodes["disease"] == event["disease"]
        )

        if str(event["district"]).upper() == "ALL":
            district_match = pd.Series(
                True, index=episodes.index
            )
        else:
            district_match = (
                episodes["district_code"].astype(str)
                == str(event["district"])
            )

        date_match = (
            (episodes["episode_start"] <= end)
            & (episodes["episode_end"] >= start)
        )

        matches = episodes.index[
            disease_match & district_match & date_match
        ].tolist()

        detected = len(matches) > 0

        matched_indices.update(matches)
        event_detected.append(detected)

    total_events = len(events)
    detected_events = sum(event_detected)

    total_episodes = len(episodes)
    matched_episodes = len(matched_indices)

    recall = (
        detected_events / total_events
        if total_events else 0
    )

    precision = (
        matched_episodes / total_episodes
        if total_episodes else 0
    )

    return {
        "alert_episodes": total_episodes,
        "matched_episodes": matched_episodes,
        "unmatched_episodes": total_episodes - matched_episodes,
        "episode_precision": precision,
        "detected_events": detected_events,
        "missed_events": total_events - detected_events,
        "event_recall": recall
    }


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading existing improved detection scores...")

    scores = pd.read_parquet(SCORES_DIR)

    scores["week_start"] = pd.to_datetime(
        scores["week_start"]
    )

    print(f"Loaded weekly scores: {len(scores):,}")

    print("Loading ground truth...")

    events = pd.read_csv(GROUND_TRUTH)

    events["start_date"] = pd.to_datetime(events["start_date"])
    events["end_date"] = pd.to_datetime(events["end_date"])

    results = []

    for multiplier in THRESHOLDS:

        print(f"\nTesting seasonal multiplier: {multiplier}")

        data = scores.copy()

        seasonal_threshold = (
            data["seasonal_mean"]
            + multiplier * data["seasonal_stddev"].fillna(0)
        )

        data["seasonal_alert"] = (
            (data["seasonal_baseline_years"] >= MIN_SEASONAL_YEARS)
            & (data["total_cases"] >= MIN_CASES)
            & (data["total_cases"] > seasonal_threshold)
        )

        data["is_outbreak_alert"] = (
            data["recent_alert"].fillna(False)
            | data["seasonal_alert"]
        )

        alerts = data[data["is_outbreak_alert"]].copy()

        episodes = create_episodes(alerts)

        metrics = evaluate(episodes, events)

        metrics["seasonal_multiplier"] = multiplier
        metrics["alert_weeks"] = len(alerts)

        results.append(metrics)

        episodes.to_csv(
            OUTPUT_DIR / f"episodes_{multiplier:.1f}.csv",
            index=False
        )

        print(f"Alert weeks: {len(alerts):,}")
        print(f"Alert episodes: {metrics['alert_episodes']:,}")
        print(f"Unmatched episodes: {metrics['unmatched_episodes']:,}")
        print(f"Episode precision: {metrics['episode_precision']:.4f}")
        print(f"Detected events: {metrics['detected_events']}/{len(events)}")
        print(f"Event recall: {metrics['event_recall']:.4f}")

    comparison = pd.DataFrame(results)

    comparison = comparison[
        [
            "seasonal_multiplier",
            "alert_weeks",
            "alert_episodes",
            "matched_episodes",
            "unmatched_episodes",
            "episode_precision",
            "detected_events",
            "missed_events",
            "event_recall"
        ]
    ]

    comparison.to_csv(
        OUTPUT_DIR / "threshold_comparison.csv",
        index=False
    )

    print("\n========== THRESHOLD COMPARISON ==========")

    print(comparison.to_string(index=False))

    print(f"\nResults saved in: {OUTPUT_DIR}")
    print("\nTHRESHOLD TUNING COMPLETED!")


if __name__ == "__main__":
    main()
