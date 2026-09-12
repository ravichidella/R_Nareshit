import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ==================================================
# 1. LOAD DATA
# ==================================================

df = pd.read_csv("telecom_churn_data.csv")


# ==================================================
# 2. REMOVE ROWS WITH MISSING TARGET
# ==================================================

df = df.dropna(subset=["churn"])


# ==================================================
# 3. REMOVE IDENTIFIER COLUMNS
# ==================================================

df = df.drop(
    columns=["customer_id", "phone_no"],
    errors="ignore"
)


# ==================================================
# 4. SEPARATE FEATURES AND TARGET
# ==================================================

X = df.drop(columns=["churn"])

y = df["churn"].astype(int)


# ==================================================
# 5. IDENTIFY COLUMN TYPES
# ==================================================

categorical_columns = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()


print("\nCategorical Columns:")
print(categorical_columns)

print("\nNumerical Columns:")
print(numerical_columns)


# ==================================================
# 6. NUMERICAL PIPELINE
# ==================================================

numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# ==================================================
# 7. CATEGORICAL PIPELINE
# ==================================================

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


# ==================================================
# 8. COMBINE PREPROCESSING
# ==================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numerical_pipeline,
            numerical_columns
        ),
        (
            "cat",
            categorical_pipeline,
            categorical_columns
        )
    ]
)


# ==================================================
# 9. TRAIN / TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining rows:", X_train.shape[0])
print("Testing rows:", X_test.shape[0])


# ==================================================
# 10. CREATE MODELS
# ==================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
}


# ==================================================
# 11. TRAIN AND EVALUATE
# ==================================================

results = []


for model_name, model in models.items():

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    # Train
    pipeline.fit(
        X_train,
        y_train
    )

    # Predict
    y_pred = pipeline.predict(
        X_test
    )

    # Probability
    y_probability = pipeline.predict_proba(
        X_test
    )[:, 1]

    # Metrics
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": roc_auc
    })


# ==================================================
# 12. DISPLAY RESULTS
# ==================================================

results_df = pd.DataFrame(results)

print("\n================ MODEL RESULTS ================\n")

print(
    results_df.to_string(
        index=False
    )
)


print(
    results_df.to_string(
        index=False
    )
)

# Return results for Streamlit
def get_model_results():
    return results_df

# ==================================================
# BEST MODEL FOR PREDICTION
# ==================================================

best_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            )
        )
    ]
)

# Train the final model using all available
# training records with known churn
best_model.fit(X, y)


def predict_churn(customer_data):

    prediction = best_model.predict(
        customer_data
    )[0]

    probability = best_model.predict_proba(
        customer_data
    )[0][1]

    return prediction, probability

