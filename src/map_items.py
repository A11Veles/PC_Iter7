import pandas as pd
import os

def map_products_to_gpc(products_file, gpc_file, output_file='product_gpc_mapping.csv'):
    
    print("Loading product information dataset...")
    products_df = pd.read_csv(products_file, sep=';')
    print(f"Loaded {len(products_df)} products")
    
    print("Loading GPC categories dataset...")
    # Handle both CSV and Excel files
    if gpc_file.endswith('.xlsx'):
        gpc_df = pd.read_excel(gpc_file)
    else:
        gpc_df = pd.read_csv(gpc_file)
    print(f"Loaded {len(gpc_df)} GPC records")
    
    # Get unique brick codes and their corresponding GPC info
    print("Processing GPC data to get unique brick codes...")
    gpc_unique = gpc_df.drop_duplicates(subset=['BrickCode'])
    print(f"Found {len(gpc_unique)} unique brick codes in GPC data")
    
    # Merge products with GPC data on brick code
    print("Mapping products to GPC categories...")
    merged_df = products_df.merge(
        gpc_unique[['BrickCode', 'SegmentTitle', 'FamilyTitle', 'ClassTitle', 'BrickTitle']], 
        left_on='brick_code', 
        right_on='BrickCode', 
        how='left'
    )
    
    # Check for unmapped products
    unmapped = merged_df[merged_df['BrickTitle'].isna()]
    if len(unmapped) > 0:
        print(f"Warning: {len(unmapped)} products could not be mapped to GPC categories")
        print("Unmapped brick codes:", unmapped['brick_code'].unique())
    
    # Create output dataframe with required columns
    output_df = pd.DataFrame({
        'Name': merged_df['productName'],
        'SegmentTitle': merged_df['SegmentTitle'],
        'FamilyTitle': merged_df['FamilyTitle'],
        'ClassTitle': merged_df['ClassTitle'],
        'BrickTitle': merged_df['BrickTitle']
    })
    
    # Save to CSV
    output_df.to_csv(output_file, index=False)
    print(f"Output saved to {output_file}")
    print(f"Successfully mapped {len(output_df)} products")
    
    # Display sample results
    print("\nSample results:")
    print(output_df.head(10))
    
    return output_df

# Main execution
if __name__ == "__main__":
    # File paths from project root directory
    products_file = "data/product-information.csv"
    gpc_file = "data/GPCMay25.xlsx"
    output_file = "data/product_gpc_mapping.csv"
    
    # Check if files exist
    if not os.path.exists(products_file):
        print(f"Error: {products_file} not found")
        exit(1)
    if not os.path.exists(gpc_file):
        print(f"Error: {gpc_file} not found")
        exit(1)
    
    # Run the mapping
    result_df = map_products_to_gpc(products_file, gpc_file, output_file)