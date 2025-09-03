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
    print("Loading GPC segments...")
    gpc_query = """
    SELECT *
    FROM demo_user.gpc_orig
    SAMPLE 20
        """
    gpc_tdf = td_db.execute_query(gpc_query)
    gpc_df = pd.DataFrame(gpc_tdf)
    gpc_df.to_csv("sample.csv")
    


if __name__ == "__main__":
    main()