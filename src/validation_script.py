import pandas as pd
import os
import re

def normalize_text(text):
    """
    Normalize text for comparison by handling various text differences
    """
    if pd.isna(text) or text == '':
        return ''
    
    # Convert to string and basic cleanup
    text = str(text).strip()
    
    # Convert to lowercase
    text = text.lower()
    
    # Replace common variations
    text = text.replace('&', 'and')
    text = text.replace('+', 'and')
    text = text.replace('/', ' ')
    text = text.replace('-', ' ')
    text = text.replace('_', ' ')
    
    # Remove punctuation except spaces
    text = re.sub(r'[^\w\s]', '', text)
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Final strip
    text = text.strip()
    
    return text

def validate_predictions(predicted_csv_path, gpc_excel_path, output_csv_path):
    """
    Validate predicted classifications against GPC reference data
    """

    # Load the data
    print("Loading data files...")
    try:
        # Load predicted data
        predicted_df = pd.read_csv(predicted_csv_path)
        print(f"Loaded predicted data: {predicted_df.shape[0]} rows")

        # Load GPC reference data
        gpc_df = pd.read_excel(gpc_excel_path)
        print(f"Loaded GPC reference data: {gpc_df.shape[0]} rows")

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # Display column info
    print(f"\nPredicted data columns: {list(predicted_df.columns)}")
    print(f"GPC data columns: {list(gpc_df.columns)}")

    # Create a lookup dictionary from GPC data for faster matching
    print("\nCreating GPC lookup dictionary...")

    # Group GPC data by BrickTitle and get unique combinations
    gpc_lookup = {}
    brick_lookup = {}  # For normalized brick matching
    
    for _, row in gpc_df.iterrows():
        brick_title = row['BrickTitle']
        if pd.notna(brick_title):
            brick_title_clean = str(brick_title).strip()
            brick_title_normalized = normalize_text(brick_title)
            
            # Store both original and normalized versions
            if brick_title_clean not in gpc_lookup:
                gpc_lookup[brick_title_clean] = {
                    'SegmentTitle': str(row['SegmentTitle']).strip() if pd.notna(row['SegmentTitle']) else '',
                    'FamilyTitle': str(row['FamilyTitle']).strip() if pd.notna(row['FamilyTitle']) else '',
                    'ClassTitle': str(row['ClassTitle']).strip() if pd.notna(row['ClassTitle']) else '',
                    'BrickTitle': brick_title_clean,
                    'SegmentTitle_norm': normalize_text(row['SegmentTitle']),
                    'FamilyTitle_norm': normalize_text(row['FamilyTitle']),
                    'ClassTitle_norm': normalize_text(row['ClassTitle']),
                    'BrickTitle_norm': brick_title_normalized
                }
            
            # Create normalized brick lookup for fuzzy matching
            if brick_title_normalized not in brick_lookup:
                brick_lookup[brick_title_normalized] = brick_title_clean

    print(f"Created lookup for {len(gpc_lookup)} unique brick titles")

    # Validate predictions
    print("\nValidating predictions...")
    valid_rows = []
    validation_stats = {
        'total_rows': len(predicted_df),
        'brick_found_exact': 0,
        'brick_found_normalized': 0,
        'segment_match': 0,
        'family_match': 0,
        'class_match': 0,
        'all_match': 0,
        'brick_not_found': 0
    }

    # Debug: Track some examples
    debug_examples = []
    debug_count = 0

    for idx, row in predicted_df.iterrows():
        predicted_brick = str(row['predicted_brick']).strip() if pd.notna(row['predicted_brick']) else ''
        predicted_segment = str(row['predicted_segment']).strip() if pd.notna(row['predicted_segment']) else ''
        predicted_family = str(row['predicted_family']).strip() if pd.notna(row['predicted_family']) else ''
        predicted_class = str(row['predicted_class']).strip() if pd.notna(row['predicted_class']) else ''

        # Normalize predicted values
        predicted_brick_norm = normalize_text(predicted_brick)
        predicted_segment_norm = normalize_text(predicted_segment)
        predicted_family_norm = normalize_text(predicted_family)
        predicted_class_norm = normalize_text(predicted_class)

        gpc_entry = None
        match_type = None

        # Try exact match first
        if predicted_brick in gpc_lookup:
            gpc_entry = gpc_lookup[predicted_brick]
            validation_stats['brick_found_exact'] += 1
            match_type = 'exact'
        # Try normalized match
        elif predicted_brick_norm in brick_lookup:
            original_brick = brick_lookup[predicted_brick_norm]
            gpc_entry = gpc_lookup[original_brick]
            validation_stats['brick_found_normalized'] += 1
            match_type = 'normalized'

        if gpc_entry:
            # Check matches for each level using normalized text
            segment_match = predicted_segment_norm == gpc_entry['SegmentTitle_norm']
            family_match = predicted_family_norm == gpc_entry['FamilyTitle_norm']
            class_match = predicted_class_norm == gpc_entry['ClassTitle_norm']

            if segment_match:
                validation_stats['segment_match'] += 1
            if family_match:
                validation_stats['family_match'] += 1
            if class_match:
                validation_stats['class_match'] += 1

            # Store debug example for first few mismatches
            if debug_count < 5 and not (segment_match and family_match and class_match):
                debug_examples.append({
                    'row': idx,
                    'match_type': match_type,
                    'predicted_brick': predicted_brick,
                    'predicted_segment': predicted_segment,
                    'predicted_family': predicted_family,
                    'predicted_class': predicted_class,
                    'gpc_segment': gpc_entry['SegmentTitle'],
                    'gpc_family': gpc_entry['FamilyTitle'],
                    'gpc_class': gpc_entry['ClassTitle'],
                    'segment_match': segment_match,
                    'family_match': family_match,
                    'class_match': class_match
                })
                debug_count += 1

            # If all levels match, keep this row
            if segment_match and family_match and class_match:
                validation_stats['all_match'] += 1
                valid_rows.append(row.to_dict())
        else:
            validation_stats['brick_not_found'] += 1

    # Print debug examples
    if debug_examples:
        print("\n" + "="*80)
        print("DEBUG: First few mismatched examples")
        print("="*80)
        for example in debug_examples:
            print(f"\nRow {example['row']} (Brick match: {example['match_type']}):")
            print(f"  Brick: '{example['predicted_brick']}'")
            print(f"  Predicted -> GPC:")
            print(f"    Segment: '{example['predicted_segment']}' -> '{example['gpc_segment']}' (Match: {example['segment_match']})")
            print(f"    Family:  '{example['predicted_family']}' -> '{example['gpc_family']}' (Match: {example['family_match']})")
            print(f"    Class:   '{example['predicted_class']}' -> '{example['gpc_class']}' (Match: {example['class_match']})")

    # Create validated dataset
    if valid_rows:
        validated_df = pd.DataFrame(valid_rows)
        validated_df.to_csv(output_csv_path, index=False)
        print(f"\nValidated dataset saved to: {output_csv_path}")
        print(f"Validated dataset contains {len(validated_df)} rows")
    else:
        print("\nNo valid rows found!")
        empty_df = pd.DataFrame(columns=predicted_df.columns)
        empty_df.to_csv(output_csv_path, index=False)
        print(f"Empty validated dataset saved to: {output_csv_path}")

    # Print validation statistics
    print("\n" + "="*50)
    print("VALIDATION STATISTICS")
    print("="*50)
    print(f"Total rows processed: {validation_stats['total_rows']}")
    
    total_brick_found = validation_stats['brick_found_exact'] + validation_stats['brick_found_normalized']
    print(f"Brick titles found (exact): {validation_stats['brick_found_exact']} ({validation_stats['brick_found_exact']/validation_stats['total_rows']*100:.1f}%)")
    print(f"Brick titles found (normalized): {validation_stats['brick_found_normalized']} ({validation_stats['brick_found_normalized']/validation_stats['total_rows']*100:.1f}%)")
    print(f"Total brick titles found: {total_brick_found} ({total_brick_found/validation_stats['total_rows']*100:.1f}%)")
    print(f"Brick titles not found: {validation_stats['brick_not_found']} ({validation_stats['brick_not_found']/validation_stats['total_rows']*100:.1f}%)")

    if total_brick_found > 0:
        print(f"\nOf the found bricks:")
        print(f"Segment matches: {validation_stats['segment_match']} ({validation_stats['segment_match']/total_brick_found*100:.1f}%)")
        print(f"Family matches: {validation_stats['family_match']} ({validation_stats['family_match']/total_brick_found*100:.1f}%)")
        print(f"Class matches: {validation_stats['class_match']} ({validation_stats['class_match']/total_brick_found*100:.1f}%)")
        print(f"Complete matches (all levels): {validation_stats['all_match']} ({validation_stats['all_match']/total_brick_found*100:.1f}%)")

    print(f"\nFinal validated rows: {validation_stats['all_match']} ({validation_stats['all_match']/validation_stats['total_rows']*100:.1f}% of total)")

    return validated_df if valid_rows else None

# Main execution
if __name__ == "__main__":
    # File paths
    predicted_csv = "data/local_labeled.csv"
    gpc_excel = "data/GPCMay25.xlsx"
    output_csv = "validated_locally_labeled.csv"

    # Run validation
    result = validate_predictions(predicted_csv, gpc_excel, output_csv)

    if result is not None:
        print(f"\nValidation completed successfully!")
        print(f"Output saved to: {output_csv}")
    else:
        print(f"\nValidation completed with no valid matches found.")