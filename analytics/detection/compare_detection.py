
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

GROUND_TRUTH_FILE = (
    PROJECT_ROOT / "data" / "ground_truth" / "ground_truth_outbreaks.csv"
)

ORIGINAL_ALERTS = (
    PROJECT_ROOT / "data" / "gold" / "outbreak_detection" / "alerts"
)

IMPROVED_ALERTS = (
    PROJECT_ROOT / "data" / "gold" / "outbreak_detection_improved" / "alerts"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "gold" / "detection_comparison"

TOLERANCE_DAYS = 7


def create_alert_episodes(alerts):
    """Merge consecutive weekly alerts into district-disease episodes."""

    if alerts.empty:
        return pd.DataFrame(columns=[
            "district_code",
            "district",
            "disease",
            "episode_start",
            "episode_end",
            "alert_weeks"
        ])

    alerts = alerts.copy()

    alerts["week_start"] = pd.to_datetime(alerts["week_start"])
    alerts = alerts.sort_values(
        ["district_code", "disease", "week_start"]
    )

    group_columns = ["district_code", "disease"]

    previous_week = (
        alerts.groupby(group_columns)["week_start"].shift(1)
    )

    new_episode = (
        previous_week.isna()
        | ((alerts["week_start"] - previous_week).dt.days > 7)
    )

    alerts["_episode_number"] = (
        new_episode.astype(int)
        .groupby([
            alerts["district_code"],
            alerts["disease"]
        ]).cumsum()
    )

    episode_keys = [
        "district_code",
        "district",
        "disease",
        "_episode_number"
    ]

    aggregations = {
        "episode_start": ("week_start", "min"),
        "episode_end": ("week_start", "max"),
        "alert_weeks": ("week_start", "count")
    }

    if "total_cases" in alerts.columns:
        aggregations["peak_cases"] = ("total_cases", "max")

    if "z_score" in alerts.columns:
        aggregations["max_z_score"] = ("z_score", "max")

    if "alert_reason" in alerts.columns:
        aggregations["alert_reasons"] = (
            "alert_reason",
            lambda values: "; ".join(sorted(set(values.dropna())))
        )

    episodes = (
        alerts.groupby(episode_keys)
        .agg(**aggregations)
        .reset_index()
        .drop(columns=["_episode_number"])
    )

    episodes["episode_end"] = (
        episodes["episode_end"] + pd.Timedelta(days=6)
    )

    return episodes


def evaluate_detector(alerts, events, detector_name):

    print(f"\nEvaluating {detector_name}...")

    episodes = create_alert_episodes(alerts)

    episodes["matched_outbreaks"] = ""
    event_results = []

    matched_episode_indices = set()

    for _, event in events.iterrows():

        event_start = (
            event["start_date"]
            - pd.Timedelta(days=TOLERANCE_DAYS)
        )

        event_end = (
            event["end_date"]
            + pd.Timedelta(days=TOLERANCE_DAYS)
        )

        disease_match = (
            episodes["disease"].astype(str)
            == str(event["disease"])
        )

        if str(event["district"]).upper() == "ALL":
            district_match = pd.Series(
                True,
                index=episodes.index
            )
        else:
            district_match = (
                episodes["district_code"].astype(str)
                == str(event["district"])
            )

        date_match = (
            (episodes["episode_start"] <= event_end)
            & (episodes["episode_end"] >= event_start)
        )

        matching_indices = episodes.index[
            disease_match & district_match & date_match
        ].tolist()

        detected = len(matching_indices) > 0

        for index in matching_indices:

            matched_episode_indices.add(index)

            previous = episodes.at[index, "matched_outbreaks"]

            if previous:
                episodes.at[index, "matched_outbreaks"] = (
                    previous + ";" + str(event["outbreak_id"])
                )
            else:
                episodes.at[index, "matched_outbreaks"] = (
                    str(event["outbreak_id"])
                )

        first_alert = None
        detection_delay = None

        if detected:
            first_alert = min(
                episodes.loc[matching_indices, "episode_start"]
            )

            detection_delay = max(
                0,
                (first_alert - event["start_date"]).days
            )

        event_results.append({
            "outbreak_id": event["outbreak_id"],
            "disease": event["disease"],
            "district": event["district"],
            "start_date": event["start_date"],
            "end_date": event["end_date"],
            "outbreak_type": event["outbreak_type"],
            "detected": detected,
            "matching_episodes": len(matching_indices),
            "first_alert_date": first_alert,
            "detection_delay_days": detection_delay
        })

    event_results = pd.DataFrame(event_results)

    total_events = len(event_results)
    detected_events = int(event_results["detected"].sum())

    event_recall = (
        detected_events / total_events
        if total_events else 0
    )

    total_episodes = len(episodes)
    true_positive_episodes = len(matched_episode_indices)

    false_positive_episodes = (
        total_episodes - true_positive_episodes
    )

    episode_precision = (
        true_positive_episodes / total_episodes
        if total_episodes else 0
    )

    episodes["matches_ground_truth"] = (
        episodes.index.isin(matched_episode_indices)
    )

    summary = {
        "detector": detector_name,
        "total_alert_weeks": len(alerts),
        "total_alert_episodes": total_episodes,
        "matched_episodes": true_positive_episodes,
        "unmatched_episodes": false_positive_episodes,
        "episode_precision": episode_precision,
        "total_ground_truth_events": total_events,
        "detected_events": detected_events,
        "missed_events": total_events - detected_events,
        "event_recall": event_recall
    }

    return summary, episodes, event_results


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading ground-truth events...")

    events = pd.read_csv(GROUND_TRUTH_FILE)

    events["start_date"] = pd.to_datetime(events["start_date"])
    events["end_date"] = pd.to_datetime(events["end_date"])

    detectors = {
        "Original": ORIGINAL_ALERTS,
        "Improved": IMPROVED_ALERTS
    }

    summaries = []

    for detector_name, alerts_path in detectors.items():

        print(f"\nLoading {detector_name} alerts...")

        alerts = pd.read_parquet(alerts_path)

        summary, episodes, event_results = evaluate_detector(
            alerts,
            events,
            detector_name
        )

        summaries.append(summary)

        episodes.to_csv(
            OUTPUT_DIR / f"{detector_name.lower()}_episodes.csv",
            index=False
        )

        event_results.to_csv(
            OUTPUT_DIR / f"{detector_name.lower()}_event_results.csv",
            index=False
        )

        event_results[
            ~event_results["detected"]
        ].to_csv(
            OUTPUT_DIR / f"{detector_name.lower()}_missed_events.csv",
            index=False
        )

        print(f"\n{detector_name} results:")
        print(f"Alert weeks: {summary['total_alert_weeks']:,}")
        print(f"Alert episodes: {summary['total_alert_episodes']:,}")
        print(f"Matched episodes: {summary['matched_episodes']:,}")
        print(f"Unmatched episodes: {summary['unmatched_episodes']:,}")
        print(
            f"Episode precision: "
            f"{summary['episode_precision']:.4f}"
        )
        print(
            f"Event recall: "
            f"{summary['event_recall']:.4f}"
        )
        print(f"Missed events: {summary['missed_events']}")

    comparison = pd.DataFrame(summaries)

    comparison.to_csv(
        OUTPUT_DIR / "detector_comparison.csv",
        index=False
    )

    print("\n========== FINAL DETECTOR COMPARISON ==========")

    print(
        comparison[
            [
                "detector",
                "total_alert_weeks",
                "total_alert_episodes",
                "matched_episodes",
                "unmatched_episodes",
                "episode_precision",
                "detected_events",
                "missed_events",
                "event_recall"
            ]
        ].to_string(index=False)
    )

    print("\nImportant:")
    print("- Precision is measured per alert episode.")
    print("- Recall is measured per ground-truth event.")
    print("- No combined F1 is reported because these metrics")
    print("  use different evaluation units.")
    print("- Nationwide events can match episodes in any district.")
    print("- Ground truth is used only for evaluation, never detection.")

    print(f"\nResults saved in: {OUTPUT_DIR}")
    print("\nDETECTOR COMPARISON COMPLETED!")


if __name__ == "__main__":
    main()
