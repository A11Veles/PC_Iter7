import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from modules.db import TeradataDatabase
from utils import load_tfidf_random_forest, load_logistic_regressiong

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

    print("Loading GPC mapping...")
    gpc_query = """
    SELECT ClassTitle, BrickTitle
    FROM demo_user.gpc_orig
    WHERE ClassTitle IS NOT NULL;
    """
    gpc_tdf = td_db.execute_query(gpc_query)
    gpc_df = pd.DataFrame(gpc_tdf)


    X = df["Name"]
    y = df["SegmentTitle"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = load_tfidf_random_forest()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    df.loc[X_test.index, "predicted_SegmentTitle"] = preds
    print("F1 Score for Segment Level:", f1_score(y_test, preds, average="weighted"))

    X = df["Name"]
    y = df["FamilyTitle"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    df.loc[X_test.index, "predicted_FamilyTitle"] = preds
    print("F1 Score for Family Level:", f1_score(y_test, preds, average= "weighted"))

    X = df["Name"]
    y = df["ClassTitle"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    df.loc[X_test.index, "predicted_ClassTitle"] = preds
    print("F1 Score for Class Level:", f1_score(y_test, preds, average= "weighted"))

    def predict_brick(name, predicted_class):
        possible_bricks = gpc_df.loc[gpc_df["ClassTitle"] == predicted_class, "BrickTitle"].dropna().unique()
        if len(possible_bricks) == 0:
            return None
        return model.predict([name])[0] if model.predict([name])[0] in possible_bricks else None

    df["Predicted_Brick"] = df.apply(
        lambda row: predict_brick(row["Name"], row["predicted_ClassTitle"]), axis=1
    )

    test_df = df[df.index.isin(X_test.index)].copy()
    test_df.to_csv("test_RF.csv", index=False)

if __name__ == "__main__":
    main()