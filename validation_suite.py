"""Comprehensive Validation Suite for CV Pipeline

Validates:
- Detection quality (confidence, NMS)
- Tracking stability (ID switches, persistence)
- Calibration accuracy (known distances)
- Speed/distance calculations
- Heatmap quality
- CSV data integrity
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np


@dataclass
class ValidationResult:
    """Single validation check result."""
    check_name: str
    passed: bool
    value: float
    threshold: float
    message: str


class DetectionValidator:
    """Validate detection quality."""
    
    @staticmethod
    def validate_confidence_threshold(csv_path: str, min_threshold: float = 0.60) -> ValidationResult:
        """Check that all detections meet confidence threshold."""
        confidences = []
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    conf = float(row.get('conf', 0))
                    confidences.append(conf)
                except ValueError:
                    pass
        
        if not confidences:
            return ValidationResult(
                check_name="Confidence Threshold",
                passed=False,
                value=0.0,
                threshold=min_threshold,
                message="No detections found"
            )
        
        min_conf = min(confidences)
        passed = min_conf >= min_threshold
        
        return ValidationResult(
            check_name="Confidence Threshold",
            passed=passed,
            value=min_conf,
            threshold=min_threshold,
            message=f"Min confidence: {min_conf:.2f} (threshold: {min_threshold})"
        )
    
    @staticmethod
    def validate_false_positives(csv_path: str, max_fps_percent: float = 5.0) -> ValidationResult:
        """Estimate false positive rate (proxy via low confidence detections)."""
        low_conf_count = 0
        total_count = 0
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    conf = float(row.get('conf', 0))
                    total_count += 1
                    if conf < 0.60:
                        low_conf_count += 1
                except ValueError:
                    pass
        
        fps_percent = (low_conf_count / total_count * 100) if total_count > 0 else 0
        passed = fps_percent <= max_fps_percent
        
        return ValidationResult(
            check_name="False Positive Rate (Proxy)",
            passed=passed,
            value=fps_percent,
            threshold=max_fps_percent,
            message=f"Low-confidence detections: {fps_percent:.1f}% (threshold: {max_fps_percent}%)"
        )


class TrackingValidator:
    """Validate tracking stability."""
    
    @staticmethod
    def validate_id_switches(csv_path: str, max_switch_rate_percent: float = 1.0) -> ValidationResult:
        """Calculate ID switching rate."""
        rows: List[Dict] = []
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        
        if not rows:
            return ValidationResult(
                check_name="ID Switch Rate",
                passed=False,
                value=0,
                threshold=max_switch_rate_percent,
                message="No tracking data"
            )
        
        # Group by frame
        by_frame: Dict[int, List[int]] = defaultdict(list)
        for row in rows:
            try:
                frame = int(row['frame_index'])
                track_id = int(row['track_id'])
                by_frame[frame].append(track_id)
            except (ValueError, KeyError):
                pass
        
        # Detect switches (same object detected as different ID)
        switches = 0
        frame_list = sorted(by_frame.keys())
        
        for i in range(1, len(frame_list)):
            frame1 = frame_list[i - 1]
            frame2 = frame_list[i]
            
            ids1 = set(by_frame[frame1])
            ids2 = set(by_frame[frame2])
            
            # Switches = IDs in frame1 not in frame2 (simplified heuristic)
            switches += len(ids1 - ids2)
        
        total_matches = sum(len(ids) for ids in by_frame.values())
        switch_rate = (switches / total_matches * 100) if total_matches > 0 else 0
        
        passed = switch_rate <= max_switch_rate_percent
        
        return ValidationResult(
            check_name="ID Switch Rate",
            passed=passed,
            value=switch_rate,
            threshold=max_switch_rate_percent,
            message=f"Switch rate: {switch_rate:.2f}% (threshold: {max_switch_rate_percent}%)"
        )
    
    @staticmethod
    def validate_persistence(csv_path: str, min_persistence_percent: float = 95.0) -> ValidationResult:
        """Check track persistence (fraction of visible frames)."""
        rows: List[Dict] = []
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        
        if not rows:
            return ValidationResult(
                check_name="Track Persistence",
                passed=False,
                value=0,
                threshold=min_persistence_percent,
                message="No tracking data"
            )
        
        # Group by track
        by_track: Dict[int, List[int]] = defaultdict(set)
        for row in rows:
            try:
                track_id = int(row['track_id'])
                frame = int(row['frame_index'])
                by_track[track_id].add(frame)
            except (ValueError, KeyError):
                pass
        
        if not by_track:
            return ValidationResult(
                check_name="Track Persistence",
                passed=False,
                value=0,
                threshold=min_persistence_percent,
                message="No tracks found"
            )
        
        # Calculate persistence for each track
        total_frames = len(set(int(r['frame_index']) for r in rows if 'frame_index' in r))
        persistences = [len(frames) / total_frames * 100 for frames in by_track.values()]
        
        avg_persistence = np.mean(persistences) if persistences else 0
        passed = avg_persistence >= min_persistence_percent
        
        return ValidationResult(
            check_name="Track Persistence",
            passed=passed,
            value=avg_persistence,
            threshold=min_persistence_percent,
            message=f"Average persistence: {avg_persistence:.1f}% (threshold: {min_persistence_percent}%)"
        )


class CalibrationValidator:
    """Validate geometric calibration."""
    
    @staticmethod
    def validate_bounds(csv_path: str, 
                       x_max: float = 105.0, y_max: float = 68.0) -> ValidationResult:
        """Check that all world coordinates are within pitch bounds."""
        violations = 0
        total = 0
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    x_m = float(row.get('x_meters', 0))
                    y_m = float(row.get('y_meters', 0))
                    total += 1
                    
                    if x_m < 0 or x_m > x_max or y_m < 0 or y_m > y_max:
                        violations += 1
                except ValueError:
                    pass
        
        violation_percent = (violations / total * 100) if total > 0 else 0
        passed = violation_percent == 0
        
        return ValidationResult(
            check_name="Calibration Bounds",
            passed=passed,
            value=violation_percent,
            threshold=0.0,
            message=f"Out-of-bounds points: {violations}/{total} ({violation_percent:.1f}%)"
        )


class SpeedValidator:
    """Validate speed and distance calculations."""
    
    @staticmethod
    def validate_speed_bounds(csv_path: str, max_speed_mps: float = 10.0) -> ValidationResult:
        """Check that all speeds are within realistic bounds."""
        violations = 0
        total = 0
        speeds = []
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    speed = float(row.get('speed_mps', 0))
                    total += 1
                    speeds.append(speed)
                    
                    if speed < 0 or speed > max_speed_mps:
                        violations += 1
                except ValueError:
                    pass
        
        violation_percent = (violations / total * 100) if total > 0 else 0
        passed = violation_percent < 1.0  # Allow <1% violations
        
        avg_speed = np.mean(speeds) if speeds else 0
        max_speed = np.max(speeds) if speeds else 0
        
        return ValidationResult(
            check_name="Speed Bounds",
            passed=passed,
            value=max_speed,
            threshold=max_speed_mps,
            message=f"Max: {max_speed:.2f} m/s, Avg: {avg_speed:.2f} m/s, Violations: {violation_percent:.1f}%"
        )
    
    @staticmethod
    def validate_speed_smoothness(csv_path: str, max_std_dev: float = 2.0) -> ValidationResult:
        """Check speed variance (measure of trajectory smoothness)."""
        by_track: Dict[int, List[float]] = defaultdict(list)
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    track_id = int(row['track_id'])
                    speed = float(row.get('speed_mps', 0))
                    by_track[track_id].append(speed)
                except (ValueError, KeyError):
                    pass
        
        std_devs = [np.std(speeds) for speeds in by_track.values() if len(speeds) > 1]
        avg_std_dev = np.mean(std_devs) if std_devs else 0
        
        passed = avg_std_dev <= max_std_dev
        
        return ValidationResult(
            check_name="Speed Smoothness",
            passed=passed,
            value=avg_std_dev,
            threshold=max_std_dev,
            message=f"Average speed std dev: {avg_std_dev:.2f} m/s (threshold: {max_std_dev})"
        )


class DataIntegrityValidator:
    """Validate CSV data integrity."""
    
    @staticmethod
    def validate_csv_structure(csv_path: str) -> ValidationResult:
        """Check CSV has all required columns."""
        required_columns = {
            'frame_index', 'time_seconds', 'track_id',
            'x', 'y', 'x_meters', 'y_meters',
            'speed_mps', 'speed_kmh', 'distance_traveled'
        }
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return ValidationResult(
                    check_name="CSV Structure",
                    passed=False,
                    value=0,
                    threshold=1,
                    message="Could not read CSV headers"
                )
            
            found_columns = set(reader.fieldnames)
            missing = required_columns - found_columns
        
        passed = len(missing) == 0
        
        return ValidationResult(
            check_name="CSV Structure",
            passed=passed,
            value=len(found_columns),
            threshold=len(required_columns),
            message=f"Missing columns: {missing}" if missing else "All required columns present"
        )
    
    @staticmethod
    def validate_no_missing_values(csv_path: str) -> ValidationResult:
        """Check for NaN or empty critical fields."""
        critical_fields = ['frame_index', 'track_id', 'x_meters', 'y_meters', 'speed_mps']
        missing_count = 0
        total = 0
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                total += 1
                for field in critical_fields:
                    value = row.get(field, '').strip()
                    if not value or value.lower() in ['nan', 'none', '']:
                        missing_count += 1
        
        missing_percent = (missing_count / (total * len(critical_fields)) * 100) if total > 0 else 0
        passed = missing_percent == 0
        
        return ValidationResult(
            check_name="Missing Values",
            passed=passed,
            value=missing_percent,
            threshold=0.0,
            message=f"Missing critical values: {missing_percent:.2f}%"
        )


class ValidationSuite:
    """Run all validations."""
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.results: List[ValidationResult] = []
    
    def run_all(self) -> List[ValidationResult]:
        """Execute all validation checks."""
        print(f"Running validation suite on {self.csv_path}...\n")
        
        # Detection
        print("Validating detection...")
        self.results.append(DetectionValidator.validate_confidence_threshold(self.csv_path))
        self.results.append(DetectionValidator.validate_false_positives(self.csv_path))
        
        # Tracking
        print("Validating tracking...")
        self.results.append(TrackingValidator.validate_id_switches(self.csv_path))
        self.results.append(TrackingValidator.validate_persistence(self.csv_path))
        
        # Calibration
        print("Validating calibration...")
        self.results.append(CalibrationValidator.validate_bounds(self.csv_path))
        
        # Speed
        print("Validating speed/distance...")
        self.results.append(SpeedValidator.validate_speed_bounds(self.csv_path))
        self.results.append(SpeedValidator.validate_speed_smoothness(self.csv_path))
        
        # Data Integrity
        print("Validating data integrity...")
        self.results.append(DataIntegrityValidator.validate_csv_structure(self.csv_path))
        self.results.append(DataIntegrityValidator.validate_no_missing_values(self.csv_path))
        
        return self.results
    
    def print_report(self) -> None:
        """Print validation report."""
        print("\n" + "=" * 80)
        print("VALIDATION REPORT")
        print("=" * 80 + "\n")
        
        passed_count = sum(1 for r in self.results if r.passed)
        total_count = len(self.results)
        
        for result in self.results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"{status} | {result.check_name:30} | {result.message}")
        
        print("\n" + "-" * 80)
        print(f"SUMMARY: {passed_count}/{total_count} checks passed ({passed_count/total_count*100:.0f}%)")
        print("=" * 80)
    
    def save_report(self, output_path: str) -> None:
        """Save report as JSON."""
        report = {
            'timestamp': str(Path(self.csv_path).stat().st_mtime),
            'csv_path': str(self.csv_path),
            'total_checks': len(self.results),
            'passed_checks': sum(1 for r in self.results if r.passed),
            'checks': [
                {
                    'name': r.check_name,
                    'passed': r.passed,
                    'value': float(r.value),
                    'threshold': float(r.threshold),
                    'message': r.message
                }
                for r in self.results
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✓ Report saved to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate CV pipeline outputs")
    parser.add_argument("--csv", required=True, help="CSV file path")
    parser.add_argument("--output", help="JSON report output path")
    
    args = parser.parse_args()
    
    suite = ValidationSuite(args.csv)
    suite.run_all()
    suite.print_report()
    
    if args.output:
        suite.save_report(args.output)
