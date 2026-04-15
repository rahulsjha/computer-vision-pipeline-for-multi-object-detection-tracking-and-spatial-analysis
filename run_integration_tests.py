#!/usr/bin/env python3
"""
End-to-end integration test runner.
Tests all improvements are working correctly with the full pipeline.
"""

import subprocess
import sys
import os
import json
from pathlib import Path
from datetime import datetime

class IntegrationTester:
    def __init__(self):
        self.results = []
        self.start_time = datetime.now()
        
    def print_header(self, title):
        """Print a formatted header."""
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}\n")
    
    def print_step(self, step_num, title, description=""):
        """Print a step header."""
        print(f"\n[Step {step_num}] {title}")
        if description:
            print(f"  {description}")
    
    def run_test(self, name, command, expected_output_file=None):
        """Run a test and check results."""
        print(f"\n  Running: {' '.join(command[:3])}...")
        
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        
        passed = result.returncode == 0
        if expected_output_file:
            passed = passed and os.path.exists(expected_output_file)
        
        status = "✓ PASS" if passed else "✗ FAIL"
        self.results.append({
            "test": name,
            "passed": passed,
            "returncode": result.returncode,
            "stdout_lines": len(result.stdout.splitlines()),
            "stderr_lines": len(result.stderr.splitlines()),
        })
        
        print(f"  {status} (exit code: {result.returncode})")
        
        if not passed and result.stderr:
            print(f"\n  Error output:")
            for line in result.stderr.splitlines()[:5]:
                print(f"    {line}")
        
        return passed
    
    def check_output_quality(self, csv_file):
        """Check the quality of output CSV."""
        if not os.path.exists(csv_file):
            print(f"  ✗ CSV file not found: {csv_file}")
            return False
        
        try:
            import csv
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            # Check required columns
            required_columns = [
                "frame_index", "track_id", "x_meters", "y_meters",
                "speed_mps", "speed_kmh", "confidence_score"
            ]
            
            missing_columns = [col for col in required_columns if col not in reader.fieldnames]
            
            if missing_columns:
                print(f"  ⚠ Missing columns: {missing_columns}")
                return False
            
            # Check data integrity
            frame_indices = {}
            for row in rows:
                frame = int(row["frame_index"])
                track_id = int(row["track_id"])
                frame_indices[track_id] = frame_indices.get(track_id, 0) + 1
            
            print(f"  ✓ CSV has {len(rows)} rows, {len(frame_indices)} unique tracks")
            
            # Check for smoothed columns if applicable
            if "x_meters_smooth" in reader.fieldnames:
                print(f"  ✓ Smoothed trajectories detected (x_meters_smooth column present)")
            
            return True
        
        except Exception as e:
            print(f"  ✗ Error reading CSV: {e}")
            return False
    
    def check_validation_report(self, report_file):
        """Check validation report."""
        if not os.path.exists(report_file):
            print(f"  ⓘ Validation report not found (validation may have been skipped)")
            return True
        
        try:
            with open(report_file, 'r') as f:
                report = json.load(f)
            
            num_checks = len(report.get("checks", []))
            num_passed = sum(1 for check in report.get("checks", []) if check.get("passed"))
            
            print(f"  ✓ Validation report: {num_passed}/{num_checks} checks passed")
            
            # Show any failed checks
            failed_checks = [c for c in report.get("checks", []) if not c.get("passed")]
            if failed_checks:
                print(f"  ⚠ {len(failed_checks)} failed checks:")
                for check in failed_checks[:3]:
                    print(f"    - {check.get('check_name', 'unknown')}")
            
            return True
        
        except Exception as e:
            print(f"  ✗ Error reading validation report: {e}")
            return False
    
    def run_all_tests(self):
        """Run all integration tests."""
        self.print_header("🚀 INTEGRATION TEST SUITE")
        
        # Test 1: Validate integration
        self.print_step(1, "Validation Check", "Verifying all modules and dependencies")
        test1_pass = self.run_test(
            "validate_integration",
            [sys.executable, "validate_integration.py"]
        )
        
        if not test1_pass:
            print("\n  ⚠️  Validation failed. Please fix issues before running pipeline.")
            sys.exit(1)
        
        # Test 2: Run pipeline with basic enhancements
        self.print_step(2, "Basic Pipeline Test", 
                       "Running with detection enhancement only")
        test2_pass = self.run_test(
            "basic_pipeline",
            [
                sys.executable, "track_cv.py",
                "--source", "output/clip_5min.mp4",
                "--output-video", "output/test_basic.mp4",
                "--output-csv", "output/test_basic.csv",
                "--use-enhanced-detection",
                "--max-frames", "100",  # Test with 100 frames first
            ],
            "output/test_basic.csv"
        )
        
        # Test 3: Run with calibration
        self.print_step(3, "Calibration Test", 
                       "Running with homography-based calibration")
        test3_pass = self.run_test(
            "calibration_pipeline",
            [
                sys.executable, "track_cv.py",
                "--source", "output/clip_5min.mp4",
                "--output-video", "output/test_calib.mp4",
                "--output-csv", "output/test_calib.csv",
                "--use-enhanced-detection",
                "--use-homography-calibration",
                "--max-frames", "100",
            ],
            "output/test_calib.csv"
        )
        
        # Test 4: Full pipeline with all enhancements
        self.print_step(4, "Full Integration Test",
                       "Running with all enhancements enabled")
        test4_pass = self.run_test(
            "full_pipeline",
            [
                sys.executable, "track_cv.py",
                "--source", "output/clip_5min.mp4",
                "--output-video", "output/test_full.mp4",
                "--output-csv", "output/test_full.csv",
                "--use-enhanced-detection",
                "--use-homography-calibration",
                "--smooth-trajectories",
                "--validate-output",
                "--max-frames", "100",
            ],
            "output/test_full.csv"
        )
        
        # Test 5: Check output quality
        self.print_step(5, "Output Quality Check",
                       "Verifying generated output files")
        
        if os.path.exists("output/test_full.csv"):
            print("\n  Checking CSV quality...")
            quality_pass = self.check_output_quality("output/test_full.csv")
        else:
            quality_pass = False
        
        if os.path.exists("output/test_full_validation.json"):
            print("\n  Checking validation report...")
            report_pass = self.check_validation_report("output/test_full_validation.json")
        else:
            report_pass = True  # Optional
        
        # Summary
        self.print_header("📊 TEST SUMMARY")
        
        total_tests = len(self.results) + 2
        passed_tests = sum(1 for r in self.results if r["passed"]) + (1 if quality_pass else 0)
        
        print(f"  Tests Passed: {passed_tests}/{total_tests}")
        print()
        
        for result in self.results:
            status = "✓" if result["passed"] else "✗"
            print(f"  {status} {result['test']}")
        
        if quality_pass:
            print(f"  ✓ Output Quality Check")
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        print(f"\n  Total Time: {elapsed:.1f} seconds")
        
        # Recommendations
        if passed_tests == total_tests:
            print("\n✅ ALL TESTS PASSED!")
            print("\nYou can now run the full production pipeline:")
            print("  python run_integrated_pipeline.py")
            return 0
        else:
            print(f"\n⚠️  {total_tests - passed_tests} test(s) failed")
            print("\nCommon issues:")
            print("  • Ensure video file exists: output/clip_5min.mp4")
            print("  • Check all module files are in the project root")
            print("  • Run: pip install -r requirement.txt")
            print("  • Check for disk space in /output directory")
            return 1

def main():
    tester = IntegrationTester()
    return tester.run_all_tests()

if __name__ == "__main__":
    sys.exit(main())
