
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ALERTS_DIR = PROJECT_ROOT / "data" / "gold" / "outbreak_detection" / "alerts"
GROUND_TRUTH_FILE = PROJECT_ROOT / "data" / "ground_truth" / "ground_truth_outbreaks.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "gold" / "detection_evaluation"

TOLERANCE_DAYS = 7


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading detected alerts...")
    alerts = pd.read_parquet(ALERTS_DIR)

    print("Loading ground-truth outbreaks...")
    events = pd.read_csv(GROUND_TRUTH_FILE)

    alerts["week_start"] = pd.to_datetime(alerts["week_start"])
    alerts["week_end"] = alerts["week_start"] + pd.Timedelta(days=6)

    events["start_date"] = pd.to_datetime(events["start_date"])
    events["end_date"] = pd.to_datetime(events["end_date"])

    # Match each alert against the known outbreak events.
    alert_matched = [False] * len(alerts)
    event_results = []

    for event_index, event in events.iterrows():
        event_start = event["start_date"] - pd.Timedelta(
            days=TOLERANCE_DAYS
        )
        event_end = event["end_date"] + pd.Timedelta(
            days=TOLERANCE_DAYS
        )

        # Nationwide event: any district, but only the specified disease.
        if event["district"] == "ALL":
            district_match = pd.Series(True, index=alerts.index)
        else:
            district_match = (
                alerts["district_code"].astype(str)
                == str(event["district"])
            )

        disease_match = (
            alerts["disease"].astype(str)
            == str(event["disease"])
        )

        # A weekly alert matches if its week overlaps the tolerated event period.
        date_match = (
            (alerts["week_start"] <= event_end)
            & (alerts["week_end"] >= event_start)
        )

        matching_indices = alerts.index[
            district_match & disease_match & date_match
        ].tolist()

        detected = len(matching_indices) > 0

        for index in matching_indices:
            alert_matched[index] = True

        event_results.append({
            "outbreak_id": event["outbreak_id"],
            "disease": event["disease"],
            "district": event["district"],
            "start_date": event["start_date"].date(),
            "end_date": event["end_date"].date(),
            "severity": event["severity"],
            "outbreak_type": event["outbreak_type"],
            "detected": detected,
            "matching_alert_weeks": len(matching_indices)
        })

    # Event-level evaluation.
    event_results = pd.DataFrame(event_results)

    total_events = len(event_results)
    detected_events = int(event_results["detected"].sum())
    missed_events = total_events - detected_events

    event_recall = (
        detected_events / total_events if total_events else 0
    )

    # Alert-level evaluation.
    alerts = alerts.copy()
    alerts["matches_ground_truth"] = alert_matched

    true_positive_alerts = int(sum(alert_matched))
    total_alerts = len(alerts)
    false_positive_alerts = total_alerts - true_positive_alerts

    precision = (
        true_positive_alerts / total_alerts if total_alerts else 0
    )

    # Event recall and alert precision are different units.
    # This F1 combines them as a descriptive summary only.
    f1_score = (
        2 * precision * event_recall / (precision + event_recall)
        if precision + event_recall > 0
        else 0
    )

    # Save detailed results.
    event_results.to_csv(
        OUTPUT_DIR / "event_detection_results.csv",
        index=False
    )

    alerts.to_csv(
        OUTPUT_DIR / "alert_evaluation_results.csv",
        index=False
    )

    missed = event_results[~event_results["detected"]]
    missed.to_csv(
        OUTPUT_DIR / "missed_outbreaks.csv",
        index=False
    )

    print("\n========== OUTBREAK EVALUATION ==========")
    print(f"Ground-truth events: {total_events}")
    print(f"Detected events: {detected_events}")
    print(f"Missed events: {missed_events}")
    print(f"Event recall: {event_recall:.4f} ({event_recall * 100:.2f}%)")

    print("\n========== ALERT EVALUATION ==========")
    print(f"Total alert weeks: {total_alerts}")
    print(f"Alert weeks overlapping known events: {true_positive_alerts}")
    print(f"Alert weeks outside known events: {false_positive_alerts}")
    print(f"Alert precision: {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Descriptive combined F1: {f1_score:.4f}")

    print("\n========== EVENT RESULTS BY TYPE ==========")
    print(
        event_results.groupby("outbreak_type")["detected"]
        .agg(["count", "sum"])
        .to_string()
    )

    print("\nMissed outbreaks:")
    if missed.empty:
        print("None")
    else:
        print(
            missed[
                ["outbreak_id", "disease", "district", "start_date", "end_date"]
            ].to_string(index=False)
        )

    print(f"\nEvaluation files saved in: {OUTPUT_DIR}")
    print("\nEVALUATION COMPLETED!")


if __name__ == "__main__":
    main()
