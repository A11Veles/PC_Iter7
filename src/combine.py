import pandas as pd

product_gpc = pd.read_csv("data/product_gpc_mapping.csv")
mwpd_full = pd.read_csv("data/MWPD_FULL.csv")

mwpd_full['BrickTitle'] = ''

combined_df = pd.concat([product_gpc, mwpd_full], ignore_index=True)

combined_df.to_csv("data/combined_data.csv", index=False)

print(f"Combined dataframe shape: {combined_df.shape}")
print("Files successfully combined and saved as 'combined_data.csv'")