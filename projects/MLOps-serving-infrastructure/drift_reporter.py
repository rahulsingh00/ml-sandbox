"""
Evidently AI Drift Reporter Module
Generates DataDriftPreset and DataSummaryPreset HTML reports and extracts summary statistics.
"""

import os
from typing import Dict, Any
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset


class EvidentlyDriftReporter:
    """Uses Evidently AI to construct rich data drift and data quality audits."""

    def __init__(self, reports_dir: str = "reports"):
        """Initializes the reporter, setting the output directory for HTML reports."""
        self.reports_dir = reports_dir
        os.makedirs(self.reports_dir, exist_ok=True)

    def generate_report(self, reference_df: pd.DataFrame, current_df: pd.DataFrame, report_id: str) -> Dict[str, Any]:
        """
        Runs Evidently presets to compare baseline (reference) vs production (current) data.
        Saves the resulting dashboard snapshot as an HTML file and returns summary metrics.
        
        Args:
            reference_df: Baseline reference dataset (e.g. historical training inputs)
            current_df: Current production dataset (e.g. recent logs)
            report_id: A unique identifier for the HTML file name
            
        Returns:
            Dict containing key metrics: dataset_drift_detected (bool), share_of_drifted_features (float), etc.
        """
        # Initialize the report with both Data Drift and Data Summary presets
        report = Report(metrics=[
            DataDriftPreset(),
            DataSummaryPreset()
        ])

        # Execute metrics calculation (current first, reference second)
        snapshot = report.run(current_data=current_df, reference_data=reference_df)

        # Save HTML page for browser-based inspection
        html_filename = f"drift_report_{report_id}.html"
        html_path = os.path.join(self.reports_dir, html_filename)
        snapshot.save_html(html_path)

        # Extract structured dictionary summary from snapshot
        report_dict = snapshot.dict()
        
        summary = {
            "report_id": report_id,
            "html_path": html_path,
            "dataset_drift_detected": False,
            "number_of_features": len(current_df.columns),
            "number_of_drifted_features": 0,
            "share_of_drifted_features": 0.0,
            "drift_share_threshold": 0.5
        }

        # Safely parse the metrics from the snapshot dictionary
        try:
            for metric in report_dict.get("metrics", []):
                metric_type = metric.get("config", {}).get("type", "")
                
                # Search for DriftedColumnsCount metric
                if "DriftedColumnsCount" in metric_type:
                    val = metric.get("value", {})
                    config = metric.get("config", {})
                    
                    count = val.get("count", 0.0)
                    share = val.get("share", 0.0)
                    threshold = config.get("drift_share", 0.5)
                    
                    summary["number_of_drifted_features"] = int(count)
                    summary["share_of_drifted_features"] = float(share)
                    summary["drift_share_threshold"] = float(threshold)
                    summary["dataset_drift_detected"] = bool(share >= threshold)
                    
                    if share > 0:
                        summary["number_of_features"] = int(round(count / share))
                    break
        except Exception as e:
            summary["parsing_error"] = str(e)

        return summary


if __name__ == "__main__":
    print("=== EVIDENTLY AI DRIFT REPORTER TEST ===\n")
    
    import numpy as np
    
    rng = np.random.default_rng(42)
    
    # Baseline
    ref_data = pd.DataFrame({
        "historical_ctr": rng.normal(loc=0.015, scale=0.005, size=200),
        "bid_floor": rng.exponential(scale=1.5, size=200),
        "hour_of_day": rng.integers(0, 24, size=200)
    })
    
    # Shifted current production data (higher bid floors, lower CTR)
    curr_data = pd.DataFrame({
        "historical_ctr": rng.normal(loc=0.008, scale=0.003, size=200),
        "bid_floor": rng.exponential(scale=3.0, size=200), # Mean shift 1.5 -> 3.0
        "hour_of_day": rng.integers(0, 24, size=200)
    })
    
    reporter = EvidentlyDriftReporter(reports_dir="projects/MLOps-serving-infrastructure/reports")
    summary = reporter.generate_report(ref_data, curr_data, report_id="demo_test")
    
    print("Evidently Report Summary:")
    print(f"  HTML saved to: {summary['html_path']}")
    print(f"  Dataset Drift Detected: {summary['dataset_drift_detected']}")
    print(f"  Features analyzed: {summary['number_of_features']}")
    print(f"  Features drifted: {summary['number_of_drifted_features']} ({summary['share_of_drifted_features'] * 100:.1f}%)")
    print(f"  Drift threshold: {summary['drift_share_threshold'] * 100:.1f}%")
    
    # Cleanup reports
    if os.path.exists(summary['html_path']):
        try:
            os.remove(summary['html_path'])
            os.rmdir("projects/MLOps-serving-infrastructure/reports")
            print("Temporary reports cleaned up successfully.")
        except OSError:
            pass
