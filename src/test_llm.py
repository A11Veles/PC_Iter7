from utils import load_llm_model, load_HierarchicalGPCClassifier
from constants import LLM_CONFIG_PATH
from modules.db import TeradataDatabase
import pandas as pd

td_db = TeradataDatabase()
td_db.connect()

print("Loading sample products...")
products_query = """
        SELECT TOP 500 id, translated_name 
        FROM demo_user.full_dataset_translated_products 
    """
products_tdf = td_db.execute_query(products_query)
products_df = pd.DataFrame(products_tdf)
print(f"Loaded {len(products_df)} products")

print("Loading GPC segments...")
gpc_query = """
SELECT SegmentTitle, FamilyTitle, ClassTitle, BrickTitle
FROM demo_user.gpc_orig
    """
gpc_tdf = td_db.execute_query(gpc_query)
gpc_df = pd.DataFrame(gpc_tdf)

model = load_HierarchicalGPCClassifier(LLM_CONFIG_PATH, gpc_df)

df = model.predict_batch(products_df, "translated_name", save_interval=10)
print(df.head(10))