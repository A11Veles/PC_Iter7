import pandas as pd
import numpy as np
from typing import List, Dict, Any
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

from modules.db import TeradataDatabase
from modules.models import (
    ICFTDCBModel, ICFTDCBModelConfig,
    SentenceEmbeddingModel, SentenceEmbeddingConfig,
    LLMModel, LLMModelConfig, TFIDFCentroidModelConfig
)
from utils import load_embedding_model, load_llm_model
from constants import E5_LARGE_INSTRUCT_CONFIG_PATH, LLM_CONFIG_PATH
from modules.models import EnsembleClassifier


def main():
    td_db = TeradataDatabase()
    td_db.connect()

    print("Loading sample products...")
    products_query = """
        SELECT id, translated_name 
        FROM demo_user.translated_products_test_dataset
    """
    products_tdf = td_db.execute_query(products_query)
    products_df = pd.DataFrame(products_tdf)
    print(f"Loaded {len(products_df)} products")

    print("Loading GPC segments...")
    gpc_query = """
    SELECT SegmentTitle, MIN(SegmentDefinition) AS SegmentDefinition
    FROM demo_user.gpc_orig
    WHERE SegmentTitle IS NOT NULL 
    GROUP BY SegmentTitle;
        """
    gpc_tdf = td_db.execute_query(gpc_query)
    gpc_df = pd.DataFrame(gpc_tdf)
    gpc_df["translated_name"] = products_df["translated_name"]
    
    unique_segments = gpc_df['SegmentTitle'].unique().tolist()
    print(f"Found {len(unique_segments)} unique segments")

    icf_config = ICFTDCBModelConfig(
        k=1,
        product_id_col="id",
        product_text_col="translated_name",
        class_name_col="class_name"
    )
    tfidf_config = TFIDFCentroidModelConfig(
            k=1,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            product_id_col="id",
            product_text_col="translated_name",
            class_name_col="class_name"
        )
    embedding_model = load_embedding_model(E5_LARGE_INSTRUCT_CONFIG_PATH)
    llm_model = load_llm_model(LLM_CONFIG_PATH)

    print("Initializing ensemble classifier...")
    ensemble = EnsembleClassifier(icf_config, tfidf_config, embedding_model, llm_model)


    print("Fitting models...")
    ensemble.fit(gpc_df, "SegmentTitle")
    
    print("Running ensemble prediction...")
    results = ensemble.predict_ensemble(
        products_df,
        product_text_col="translated_name",
        voting_strategy="majority"
    )

    print("Results:")
    print(results.head(10))

    filtered_results = results[
    ~results["llm_prediction"].isin(["food beverage", "toys games"])
    ]
    results.to_csv("ensemble_predictions_majority.csv", index=False)
    filtered_results.to_csv("ensemble_predictions_majority_but_filtered.csv", index=False)

    print("\nResults saved to ensemble_predictions.csv")
        

if __name__ == "__main__":
    main()