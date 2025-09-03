import csv
import json

with open("consumer_products_dataset.json", "r", encoding="utf-8") as file:
    json_data = json.load(file)

def remove_asterisks(data):
    if isinstance(data, dict):
        return {key.replace("**", ""): remove_asterisks(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [item.replace("**", "") if isinstance(item, str) else remove_asterisks(item) for item in data]
    elif isinstance(data, str):
        return data.replace("**", "")
    return data

# Cleaned JSON data
cleaned_data = remove_asterisks(json_data)

# Flatten JSON to CSV rows
rows = []
for sector, brands in cleaned_data.items():
    for brand, products in brands.items():
        for product in products:
            rows.append([sector, brand, product])

# Write to CSV
csv_file = "converted_brand_data.csv"
with open(csv_file, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Sector", "Brand", "Product"])
    writer.writerows(rows)

print(f"CSV file '{csv_file}' has been created.")