import pandas as pd
import re

df1 = pd.read_csv("data/correctly_matched_mapped_gpc.csv")
df2 = pd.read_csv("data/MWPD_FULL.csv")

print("df1 shape:", df1.shape)
print("df1 columns:", df1.columns.tolist())
print("\ndf2 shape:", df2.shape)
print("df2 columns:", df2.columns.tolist())

df1_subset = df1[['segment', 'family', 'class', 'brick', 'product_name']].copy()

df2_subset = df2[['SegmentTitle', 'FamilyTitle', 'ClassTitle', 'Name']].copy()
df2_subset = df2_subset.rename(columns={
    'SegmentTitle': 'segment',
    'FamilyTitle': 'family', 
    'ClassTitle': 'class',
    'Name': 'product_name'
})

df2_subset['brick'] = ''

print("\ndf1_subset shape:", df1_subset.shape)
print("df2_subset shape:", df2_subset.shape)

merged_df = pd.concat([df1_subset, df2_subset], ignore_index=True)

# merged_df = merged_df.drop_duplicates().reset_index(drop=True)

merged_df.insert(0, 'id', range(len(merged_df)))

print("\nMerged dataframe shape:", merged_df.shape)
print("Merged dataframe columns:", merged_df.columns.tolist())
print("\nFirst few rows:")
print(merged_df.head())

# Text normalization function
def normalize_text(text):
    if pd.isna(text) or text == '':
        return text
    # Replace numbers and punctuation with spaces, keep only letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', str(text))
    # Convert to lowercase
    text = text.lower()
    # Remove extra spaces
    text = ' '.join(text.split())
    return text

# Apply normalization to all columns except 'id'
text_columns = ['segment', 'family', 'class', 'brick', 'product_name']
for col in text_columns:
    merged_df[col] = merged_df[col].apply(normalize_text)

print("\nAfter normalization - First few rows:")
print(merged_df.head())

merged_df.to_csv("data/full_dataset.csv", index=False)
print("\nMerged CSV file saved as 'data/full_dataset.csv'")