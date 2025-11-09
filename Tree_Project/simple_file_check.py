#!/usr/bin/env python3
"""
Simple script to check file paths and fix the dataset loader.
"""

import os
import glob

def check_paths():
    print("=== File Path Debugging ===")
    
    base_dir = "../Images"
    print(f"Base directory: {base_dir}")
    print(f"Absolute path: {os.path.abspath(base_dir)}")
    print(f"Exists: {os.path.exists(base_dir)}")
    
    if os.path.exists(base_dir):
        # List contents
        contents = os.listdir(base_dir)
        print(f"Contents: {contents}")
        
        # Check for S1 specifically
        s1_path = os.path.join(base_dir, "S1")
        print(f"\nS1 path: {s1_path}")
        print(f"S1 exists: {os.path.exists(s1_path)}")
        
        if os.path.exists(s1_path):
            s1_contents = os.listdir(s1_path)
            print(f"S1 contents: {s1_contents}")
            
            # Check s1t1
            s1t1_path = os.path.join(s1_path, "s1t1")
            print(f"\ns1t1 path: {s1t1_path}")
            print(f"s1t1 exists: {os.path.exists(s1t1_path)}")
            
            if os.path.exists(s1t1_path):
                s1t1_contents = os.listdir(s1t1_path)
                print(f"s1t1 contents: {s1t1_contents}")
                
                # Check label directory
                label_path = os.path.join(s1t1_path, "label")
                print(f"\nLabel path: {label_path}")
                print(f"Label exists: {os.path.exists(label_path)}")
                
                if os.path.exists(label_path):
                    label_files = os.listdir(label_path)
                    print(f"Label files: {label_files[:10]}")  # First 10
                    
                    # Try to read one file
                    if label_files:
                        sample_file = os.path.join(label_path, label_files[0])
                        print(f"\nSample file: {sample_file}")
                        print(f"Sample file exists: {os.path.exists(sample_file)}")
                        print(f"Sample file size: {os.path.getsize(sample_file) if os.path.exists(sample_file) else 'N/A'} bytes")
    
    print("\n=== Glob Pattern Testing ===")
    # Test different glob patterns
    patterns = [
        os.path.join(base_dir, "S*"),
        os.path.join(base_dir, "S1"),
        os.path.join(base_dir, "S1", "s*"),
        os.path.join(base_dir, "S1", "s1t1"),
        os.path.join(base_dir, "S1", "s1t1", "label"),
        os.path.join(base_dir, "S1", "s1t1", "label", "*"),
    ]
    
    for pattern in patterns:
        results = glob.glob(pattern)
        print(f"Pattern: {pattern}")
        print(f"Results: {results}")
        print()

def create_simple_loader():
    """Create a very simple version that manually finds files."""
    
    print("=== Creating Simple File List ===")
    
    base_dir = "../Images"
    all_files = []
    
    # Manually walk through the directory structure
    if os.path.exists(base_dir):
        for subject_dir in os.listdir(base_dir):
            if subject_dir.startswith('S') and not subject_dir.startswith('.'):
                subject_path = os.path.join(base_dir, subject_dir)
                if os.path.isdir(subject_path):
                    print(f"Processing {subject_dir}...")
                    
                    for timepoint_dir in os.listdir(subject_path):
                        if not timepoint_dir.startswith('.'):
                            timepoint_path = os.path.join(subject_path, timepoint_dir)
                            if os.path.isdir(timepoint_path):
                                
                                label_path = os.path.join(timepoint_path, "label")
                                if os.path.exists(label_path):
                                    
                                    for filename in os.listdir(label_path):
                                        if not filename.startswith('.'):
                                            file_path = os.path.join(label_path, filename)
                                            if os.path.isfile(file_path):
                                                
                                                # Extract subject ID
                                                try:
                                                    subid = int(subject_dir[1:])  # Remove 'S'
                                                except:
                                                    subid = 0
                                                
                                                all_files.append({
                                                    'subid': subid,
                                                    'timepoint': timepoint_dir,
                                                    'filename': filename,
                                                    'full_path': file_path,
                                                    'file_size': os.path.getsize(file_path)
                                                })
    
    print(f"Found {len(all_files)} files total")
    
    # Show some examples
    for i, file_info in enumerate(all_files[:10]):
        print(f"  {i+1}: Subject {file_info['subid']}, {file_info['timepoint']}, {file_info['filename']}, {file_info['file_size']} bytes")
    
    return all_files

if __name__ == "__main__":
    check_paths()
    files = create_simple_loader()