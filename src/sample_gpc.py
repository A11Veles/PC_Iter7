import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def create_gpc_representative_sample(df, target_sample_size=5000, min_samples_per_class=2, random_state=42):
    """
    Create a representative sample of GPC data ensuring variety across all classes.
    
    Parameters:
    - df: DataFrame with GPC data
    - target_sample_size: Target number of samples (default 5000)
    - min_samples_per_class: Minimum samples per class (default 2)
    - random_state: Random seed for reproducibility
    
    Returns:
    - sample_df: Representative sample DataFrame
    - sampling_report: Dictionary with sampling statistics
    """
    
    np.random.seed(random_state)
    
    print("🔍 Analyzing GPC dataset structure...")
    
    # Basic dataset info
    total_rows = len(df)
    unique_segments = df['SegmentCode'].nunique()
    unique_families = df['FamilyCode'].nunique() 
    unique_classes = df['ClassCode'].nunique()
    unique_bricks = df['BrickCode'].nunique()
    
    print(f"📊 Dataset Overview:")
    print(f"   Total rows: {total_rows:,}")
    print(f"   Unique Segments: {unique_segments}")
    print(f"   Unique Families: {unique_families}")
    print(f"   Unique Classes: {unique_classes}")
    print(f"   Unique Bricks: {unique_bricks}")
    
    # Analyze class distribution
    class_counts = df.groupby('ClassCode').size().sort_values(ascending=False)
    print(f"\n📈 Class Distribution:")
    print(f"   Largest class: {class_counts.iloc[0]:,} samples")
    print(f"   Smallest class: {class_counts.iloc[-1]:,} samples")
    print(f"   Median class size: {class_counts.median():.0f} samples")
    
    # Check if we can guarantee minimum samples per class
    required_min_samples = unique_classes * min_samples_per_class
    if required_min_samples > target_sample_size:
        adjusted_min = max(1, target_sample_size // unique_classes)
        print(f"⚠️  Adjusting min_samples_per_class from {min_samples_per_class} to {adjusted_min}")
        min_samples_per_class = adjusted_min
    
    print(f"\n🎯 Sampling Strategy:")
    print(f"   Target sample size: {target_sample_size:,}")
    print(f"   Min samples per class: {min_samples_per_class}")
    print(f"   Reserved for minimums: {unique_classes * min_samples_per_class:,}")
    print(f"   Available for proportional: {target_sample_size - (unique_classes * min_samples_per_class):,}")
    
    # Step 1: Ensure minimum representation per class
    sampled_dfs = []
    remaining_budget = target_sample_size
    
    for class_code in class_counts.index:
        class_data = df[df['ClassCode'] == class_code]
        class_size = len(class_data)
        
        # Take minimum required samples
        min_take = min(min_samples_per_class, class_size)
        class_sample = class_data.sample(n=min_take, random_state=random_state)
        sampled_dfs.append(class_sample)
        remaining_budget -= min_take
    
    print(f"\n✅ Phase 1 Complete: {len(sampled_dfs)} classes sampled with minimums")
    print(f"   Remaining budget: {remaining_budget:,}")
    
    # Step 2: Allocate remaining budget proportionally
    if remaining_budget > 0:
        # Calculate proportional allocation
        total_remaining = sum(max(0, class_counts[cls] - min_samples_per_class) for cls in class_counts.index)
        
        additional_samples = []
        for i, class_code in enumerate(class_counts.index):
            class_data = df[df['ClassCode'] == class_code]
            class_size = len(class_data)
            
            # Skip classes already exhausted
            available = max(0, class_size - min_samples_per_class)
            if available == 0:
                continue
                
            # Calculate proportional share
            if total_remaining > 0:
                proportion = available / total_remaining
                additional_needed = int(remaining_budget * proportion)
                
                # Don't exceed what's available
                additional_needed = min(additional_needed, available)
                
                if additional_needed > 0:
                    # Get samples not already taken
                    already_sampled = sampled_dfs[i]
                    remaining_data = class_data.drop(already_sampled.index)
                    
                    if len(remaining_data) >= additional_needed:
                        additional_sample = remaining_data.sample(n=additional_needed, random_state=random_state)
                        additional_samples.append(additional_sample)
                        remaining_budget -= additional_needed
        
        print(f"✅ Phase 2 Complete: Additional {len(additional_samples)} samples allocated proportionally")
        sampled_dfs.extend(additional_samples)
    
    # Combine all samples
    final_sample = pd.concat(sampled_dfs, ignore_index=True)
    
    # Step 3: If we're still under budget, add random samples
    if len(final_sample) < target_sample_size and len(final_sample) < total_rows:
        remaining_needed = target_sample_size - len(final_sample)
        unused_data = df.drop(final_sample.index)
        
        if len(unused_data) > 0:
            additional_random = unused_data.sample(n=min(remaining_needed, len(unused_data)), 
                                                 random_state=random_state)
            final_sample = pd.concat([final_sample, additional_random], ignore_index=True)
            print(f"✅ Phase 3 Complete: Added {len(additional_random)} random samples")
    
    # Shuffle the final sample
    final_sample = final_sample.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Generate sampling report
    sample_class_counts = final_sample.groupby('ClassCode').size()
    sample_segment_counts = final_sample.groupby('SegmentCode').size()
    
    sampling_report = {
        'original_size': total_rows,
        'sample_size': len(final_sample),
        'sample_percentage': len(final_sample) / total_rows * 100,
        'classes_represented': len(sample_class_counts),
        'classes_coverage': len(sample_class_counts) / unique_classes * 100,
        'segments_represented': len(sample_segment_counts), 
        'segments_coverage': len(sample_segment_counts) / unique_segments * 100,
        'min_class_samples': sample_class_counts.min(),
        'max_class_samples': sample_class_counts.max(),
        'avg_class_samples': sample_class_counts.mean()
    }
    
    print(f"\n🎉 Sampling Complete!")
    print(f"   Final sample size: {len(final_sample):,}")
    print(f"   Classes represented: {sampling_report['classes_represented']}/{unique_classes} ({sampling_report['classes_coverage']:.1f}%)")
    print(f"   Segments represented: {sampling_report['segments_represented']}/{unique_segments} ({sampling_report['segments_coverage']:.1f}%)")
    print(f"   Samples per class: {sampling_report['min_class_samples']} to {sampling_report['max_class_samples']} (avg: {sampling_report['avg_class_samples']:.1f})")
    
    return final_sample, sampling_report

def analyze_sample_quality(original_df, sample_df):
    """
    Analyze the quality and representativeness of the sample.
    """
    print("\n📋 Sample Quality Analysis:")
    
    # Compare distributions
    orig_segment_dist = original_df['SegmentCode'].value_counts(normalize=True).sort_index()
    sample_segment_dist = sample_df['SegmentCode'].value_counts(normalize=True).sort_index()
    
    print(f"\n🏭 Top 5 Segment Distributions:")
    print("Segment\t\tOriginal%\tSample%\t\tDifference")
    print("-" * 55)
    for segment in orig_segment_dist.head().index:
        orig_pct = orig_segment_dist.get(segment, 0) * 100
        sample_pct = sample_segment_dist.get(segment, 0) * 100
        diff = abs(orig_pct - sample_pct)
        print(f"{segment}\t{orig_pct:.1f}%\t\t{sample_pct:.1f}%\t\t{diff:.1f}%")
    
    # Attribute diversity
    attr_columns = ['AttributeCode', 'AttributeValueCode']
    for col in attr_columns:
        if col in original_df.columns and col in sample_df.columns:
            orig_unique = original_df[col].nunique()
            sample_unique = sample_df[col].nunique() 
            coverage = sample_unique / orig_unique * 100
            print(f"\n🏷️  {col} coverage: {sample_unique:,}/{orig_unique:,} ({coverage:.1f}%)")

def main():
    """
    Main function to load GPC data and create representative sample.
    """
    
    # Define file paths relative to script location
    script_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
    project_root = script_dir.parent  # Go up one level from src to p
    data_file = project_root / "data" / "GPCMay25.xlsx"
    
    print(f"📁 Loading GPC dataset from: {data_file}")
    
    # Check if file exists
    if not data_file.exists():
        print(f"❌ Error: File not found at {data_file}")
        print(f"   Current working directory: {Path.cwd()}")
        print(f"   Script directory: {script_dir}")
        print(f"   Expected data directory: {project_root / 'data'}")
        return None, None
    
    try:
        # Load the Excel file
        print("📖 Reading Excel file...")
        df = pd.read_excel(data_file)
        print(f"✅ Successfully loaded {len(df):,} rows and {len(df.columns)} columns")
        
        # Display column names to verify structure
        print(f"\n📋 Available columns:")
        for i, col in enumerate(df.columns, 1):
            print(f"   {i:2d}. {col}")
        
        # Check for required columns
        required_columns = ['SegmentCode', 'FamilyCode', 'ClassCode', 'BrickCode']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"\n⚠️  Warning: Missing required columns: {missing_columns}")
            print("   Please verify your data has the correct column names.")
            return None, None
        
        # Display sample of data
        print(f"\n👀 First 3 rows of data:")
        print(df.head(3).to_string())
        
        # Create representative sample
        print(f"\n🚀 Creating representative sample...")
        sample_df, report = create_gpc_representative_sample(
            df, 
            target_sample_size=5000, 
            min_samples_per_class=2,
            random_state=42
        )
        
        # Analyze sample quality
        analyze_sample_quality(df, sample_df)
        
        # Save the sample
        output_file = project_root / "data" / "gpc_representative_sample.csv"
        sample_df.to_csv(output_file, index=False)
        print(f"\n💾 Sample saved to: {output_file}")
        
        # Save sampling report
        report_file = project_root / "data" / "sampling_report.txt"
        with open(report_file, 'w') as f:
            f.write("GPC Representative Sampling Report\n")
            f.write("=" * 40 + "\n\n")
            for key, value in report.items():
                f.write(f"{key}: {value}\n")
        
        print(f"📊 Sampling report saved to: {report_file}")
        
        return sample_df, report
        
    except Exception as e:
        print(f"❌ Error loading or processing data: {str(e)}")
        return None, None

if __name__ == "__main__":
    sample, report = main()