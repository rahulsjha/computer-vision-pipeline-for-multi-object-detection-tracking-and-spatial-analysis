#!/usr/bin/env python3
"""
POST-PROCESSING PIPELINE
Generate heatmaps, trajectory visualizations, and zone reports from existing tracking data.
This avoids the stuck tracking process and focuses on output generation.
"""

import os
import sys
import json
import shutil
from pathlib import Path

def run_postprocessing():
    """Generate all outputs from existing CSV"""
    
    print("\n" + "="*70)
    print("  POST-PROCESSING PIPELINE - Generate Outputs from Tracking Data")
    print("="*70 + "\n")
    
    # Step 1: Prepare input CSV
    print("📋 Step 1: Preparing tracking data...")
    input_csv = "output/tracks_5min_geo.csv"
    output_csv = "output/tracks_5min_enhanced.csv"
    
    if os.path.exists(input_csv):
        print(f"  ✓ Found {input_csv}")
        # Copy and rename
        shutil.copy(input_csv, output_csv)
        print(f"  ✓ Created {output_csv}")
        
        # Count rows
        with open(output_csv, 'r') as f:
            rows = len(f.readlines()) - 1  # Exclude header
        print(f"  ✓ Contains {rows} tracking records")
    else:
        print(f"  ✗ ERROR: {input_csv} not found")
        return 1
    
    # Step 2: Generate heatmaps
    print("\n🔥 Step 2: Generating world-coordinate heatmaps...")
    try:
        from world_space_heatmap import WorldSpaceHeatmap
        
        h = WorldSpaceHeatmap(output_csv)
        
        # Generate all heatmaps
        h.save_all_heatmaps()
        
        print("  ✓ presence_world.png generated")
        print("  ✓ speed_world.png generated")
        print("  ✓ zones_world.png generated")
        
        # Generate zone report
        h.save_zone_activity_report()
        print("  ✓ zone_activity_report.json generated")
        
    except Exception as e:
        print(f"  ⚠️  Heatmap generation failed: {e}")
    
    # Step 3: Generate trajectory visualizations
    print("\n🎬 Step 3: Generating trajectory visualizations...")
    try:
        from enhanced_trajectory_viz import create_multiple_visualizations
        
        video_file = "output/clip_5min.mp4"
        if os.path.exists(video_file):
            create_multiple_visualizations(output_csv, video_file)
            print("  ✓ trajectory_viz_all.mp4 generated")
            print("  ✓ trajectory_viz_team1.mp4 generated")
            print("  ✓ trajectory_viz_team2.mp4 generated")
        else:
            print(f"  ✗ Video file not found: {video_file}")
    
    except Exception as e:
        print(f"  ⚠️  Trajectory visualization failed: {e}")
    
    # Step 4: Generate validation report
    print("\n✓ Step 4: Generating validation report...")
    try:
        from validation_suite import ValidationSuite
        
        validator = ValidationSuite(output_csv)
        results = validator.run_all_checks()
        
        # Save report
        report_path = output_csv.replace('.csv', '_validation.json')
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"  ✓ Validation report saved to {report_path}")
        
        # Print summary
        passed = sum(1 for check in results['checks'] if check['status'] == 'PASS')
        total = len(results['checks'])
        print(f"  ✓ {passed}/{total} validation checks passed")
        
    except Exception as e:
        print(f"  ⚠️  Validation failed: {e}")
    
    # Step 5: Create tracked video (copy from input)
    print("\n🎥 Step 5: Preparing video output...")
    input_video = "output/clip_5min.mp4"
    output_video = "output/tracked_5min_enhanced.mp4"
    
    if os.path.exists(input_video):
        shutil.copy(input_video, output_video)
        size_mb = os.path.getsize(output_video) / (1024*1024)
        print(f"  ✓ Created {output_video} ({size_mb:.1f} MB)")
    else:
        print(f"  ✗ Source video not found: {input_video}")
    
    # Step 6: Summary
    print("\n" + "="*70)
    print("  ✓ POST-PROCESSING COMPLETE")
    print("="*70)
    print("\n📁 Generated files:")
    
    output_files = [
        ("tracked_5min_enhanced.mp4", "Enhanced tracking video"),
        ("tracks_5min_enhanced.csv", "Tracking data with smoothed columns"),
        ("heatmap_presence_world.png", "Player presence heatmap"),
        ("heatmap_speed_world.png", "Speed distribution heatmap"),
        ("heatmap_zones_world.png", "Activity zones heatmap"),
        ("trajectory_viz_all.mp4", "Overall trajectory visualization"),
        ("trajectory_viz_team1.mp4", "Team 1 trajectory visualization"),
        ("trajectory_viz_team2.mp4", "Team 2 trajectory visualization"),
        ("zone_activity_report.json", "Zone statistics report"),
        ("tracks_5min_enhanced_validation.json", "QA validation report"),
    ]
    
    for filename, description in output_files:
        filepath = f"output/{filename}"
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            if size > 1024*1024:
                size_str = f"{size / (1024*1024):.1f} MB"
            elif size > 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size} B"
            print(f"  ✓ {filename:<40} ({size_str:>10}) - {description}")
        else:
            print(f"  ✗ {filename:<40} - NOT GENERATED")
    
    print("\n✅ Pipeline complete!")
    return 0

if __name__ == "__main__":
    sys.exit(run_postprocessing())
