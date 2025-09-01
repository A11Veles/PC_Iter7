import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from modules.db import TeradataDatabase
from utils import load_logistic_regressiong


def duplicate_minority_classes(df, levels=["SegmentTitle", "FamilyTitle", "ClassTitle", "BrickTitle"]):
    for level in levels:
        counts = df[level].value_counts()
        min_count = counts.min()
        minority_classes = counts[counts == min_count].index
        for cls in minority_classes:
            minority_rows = df[df[level] == cls]
            df = df._append(minority_rows, ignore_index=True)
    df.reset_index(drop=True, inplace=True)
    return df


def create_logistic_pipeline():
    lr_model = load_logistic_regressiong()
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        max_features=10000,
        lowercase=True,
        strip_accents='unicode',
        stop_words='english'
    )
    pipeline = Pipeline([
        ('tfidf', vectorizer),
        ('lr', lr_model)
    ])
    return pipeline


def main():
    td_db = TeradataDatabase()
    td_db.connect()

    print("Loading sample products...")
    products_query = """
    SELECT * FROM demo_user.mwpd_combined_cleaned;
    """
    products_tdf = td_db.execute_query(products_query)
    df = pd.DataFrame(products_tdf)

    df = duplicate_minority_classes(df)
    print(f"Dataset size after duplication: {len(df):,} samples")

    print("Loading GPC mapping...")
    gpc_query = """
    SELECT ClassTitle, BrickTitle
    FROM demo_user.gpc_orig
    WHERE ClassTitle IS NOT NULL;
    """
    gpc_tdf = td_db.execute_query(gpc_query)
    gpc_df = pd.DataFrame(gpc_tdf)

    pipeline = create_logistic_pipeline()
    
    print("\n SEGMENT LEVEL CLASSIFICATION")
    
    X = df["Name"]
    y = df["SegmentTitle"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    
    df.loc[X_test.index, "predicted_SegmentTitle"] = preds
    
    segment_f1 = f1_score(y_test, preds, average="weighted")
    segment_acc = accuracy_score(y_test, preds)
    
    print(f" F1 Score (Weighted): {segment_f1:.4f}")

    print("\n FAMILY LEVEL CLASSIFICATION") 
    
    X = df["Name"]
    y = df["FamilyTitle"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    
    df.loc[X_test.index, "predicted_FamilyTitle"] = preds
    
    family_f1 = f1_score(y_test, preds, average="weighted")
    family_acc = accuracy_score(y_test, preds)
    
    print(f" F1 Score (Weighted): {family_f1:.4f}")


    print("\n CLASS LEVEL CLASSIFICATION")

    
    X = df["Name"]
    y = df["ClassTitle"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    
    df.loc[X_test.index, "predicted_ClassTitle"] = preds
    
    class_f1 = f1_score(y_test, preds, average="weighted")
    class_acc = accuracy_score(y_test, preds)
    
    print(f" F1 Score (Weighted): {class_f1:.4f}")

    print("\n BRICK LEVEL PREDICTION")
    
    def predict_brick(name, predicted_class):
        possible_bricks = gpc_df.loc[
            gpc_df["ClassTitle"] == predicted_class, "BrickTitle"
        ].dropna().unique()
        
        if len(possible_bricks) == 0:
            return None
        elif len(possible_bricks) == 1:
            return possible_bricks[0]
        else:
            try:
                pred = pipeline.predict([name])[0]
                return pred if pred in possible_bricks else possible_bricks[0]
            except:
                return possible_bricks[0]

    df["predicted_BrickTitle"] = df.apply(
        lambda row: predict_brick(row["Name"], row["predicted_ClassTitle"]), 
        axis=1
    )
    
    brick_predictions = df["predicted_BrickTitle"].notna().sum()
    print(f"Successful brick predictions: {brick_predictions:,}/{len(df):,} ({brick_predictions/len(df)*100:.1f}%)")


    test_df = df[df.index.isin(X_test.index)].copy()
    test_df.to_csv("test_LR.csv", index=False)

if __name__ == "__main__":
    main()
