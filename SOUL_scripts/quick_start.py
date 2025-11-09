#!/usr/bin/env python3
"""
Simple setup script for SOUL dataset.
Assumes you're running this from Tree Project directory and OCTA is the parent.
"""

import os
import pandas as pd
import glob
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("=== SOUL Dataset Quick Setup ===")
    
    # Set paths based on your structure
    excel_file = "Infomation.xlsx.xlsx"  # Should be in Tree Project directory
    images_base = "../Images"  # OCTA/Images relative to Tree Project
    
    # Check if Excel file exists
    if not os.path.exists(excel_file):
        logger.error(f"Excel file not found: {excel_file}")
        logger.info("Please ensure 'Infomation.xlsx.xlsx' is in the Tree Project directory")
        return
    
    # Check if Images directory exists
    if not os.path.exists(images_base):
        logger.error(f"Images directory not found: {images_base}")
        logger.info("Please ensure you're running this from Tree Project directory")
        logger.info("And that OCTA/Images exists")
        return
    
    logger.info(f"✓ Found Excel file: {excel_file}")
    logger.info(f"✓ Found Images directory: {os.path.abspath(images_base)}")
    
    # Quick scan of directory structure
    subject_dirs = glob.glob(os.path.join(images_base, "S*"))
    logger.info(f"✓ Found {len(subject_dirs)} subject directories")
    
    # Check a few subjects
    if subject_dirs:
        sample_subject = subject_dirs[0]
        logger.info(f"Sample subject: {os.path.basename(sample_subject)}")
        
        # Look for timepoint directories
        timepoints = glob.glob(os.path.join(sample_subject, "st*"))
        if timepoints:
            sample_tp = timepoints[0]
            logger.info(f"Sample timepoint: {os.path.basename(sample_tp)}")
            
            # Check for label directory
            label_dir = os.path.join(sample_tp, "label")
            if os.path.exists(label_dir):
                label_files = glob.glob(os.path.join(label_dir, "*"))
                logger.info(f"✓ Label directory found with {len(label_files)} files")
                
                # Show sample files
                for i, file in enumerate(sorted(label_files)[:3]):
                    logger.info(f"  Example {i+1}: {os.path.basename(file)}")
            else:
                logger.warning("Label directory not found")
    
    # Analyze Excel file quickly
    try:
        basic_info = pd.read_excel(excel_file, sheet_name='basic_info')
        image_info = pd.read_excel(excel_file, sheet_name='image_info')
        
        logger.info(f"✓ Excel data: {len(basic_info)} subjects, {len(image_info)} images")
        
        # Show subset info
        subset_cols = [col for col in image_info.columns if col.startswith('Subset')]
        logger.info("Subset distributions:")
        for col in subset_cols:
            count = image_info[col].notna().sum()
            logger.info(f"  {col}: {count} images")
            
    except Exception as e:
        logger.error(f"Error reading Excel file: {e}")
        return
    
    # Create training command
    logger.info("\n=== TRAINING COMMAND ===")
    
    command = f"""python label_based_soul_dataset.py \\
    --excel_file "{excel_file}" \\
    --base_dir "{images_base}" \\
    --subset "Subset.1.1" \\
    --max_samples 20 \\
    --batch_size 4 \\
    --epochs 10 \\
    --lr 0.001 \\
    --hidden_dim 32 \\
    --output_dim 64 \\
    --save_dir "./soul_checkpoints" """
    
    logger.info(f"\n{command}")
    
    # Create a simple run script
    with open("run_training.sh", "w") as f:
        f.write("#!/bin/bash\n")
        f.write("echo 'Starting SOUL Geometric Tree Training...'\n")
        f.write(command.replace("\\", ""))
        f.write("\n")
    
    os.chmod("run_training.sh", 0o755)
    
    logger.info("\n=== NEXT STEPS ===")
    logger.info("1. Ensure you have the geometric tree training script in this directory")
    logger.info("2. Install requirements: pip install torch torch-geometric pandas opencv-python scikit-learn networkx tqdm")
    logger.info("3. Run: ./run_training.sh")
    logger.info("4. Or copy the command above and run it directly")
    
    logger.info("\n=== FILES NEEDED IN THIS DIRECTORY ===")
    logger.info("- label_based_soul_dataset.py (the main training script)")
    logger.info("- Infomation.xlsx.xlsx (your data file)")
    logger.info("- This setup script")
    
    logger.info(f"\nSetup complete! Ready to train on SOUL dataset.")

if __name__ == "__main__":
    main()