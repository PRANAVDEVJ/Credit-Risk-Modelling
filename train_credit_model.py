# train_credit_model.py

import pandas as pd                      # pandas: used for loading and manipulating tabular data (DataFrame)
import joblib                            # joblib: used to save/load Python objects (models, encoders)
from sklearn.model_selection import train_test_split  # split data into train and test sets
from sklearn.preprocessing import LabelEncoder         # convert categorical text labels to numeric labels
from sklearn.ensemble import ExtraTreesClassifier       # ensemble classifier (many randomized decision trees)
from sklearn.metrics import accuracy_score             # function to compute accuracy of predictions

# ---------------------------------
# 1️⃣ Load Dataset
# ---------------------------------
df = pd.read_csv("german_credit_data.csv")  # load CSV file into a DataFrame named `df`

print("✅ Data loaded successfully!")        # print confirmation to console
print(df.head())                            # print first 5 rows so you can quickly inspect columns & sample rows

# ---------------------------------
# 2️⃣ Define Features & Target
# ---------------------------------
features = ["Age", "Sex", "Job", "Housing", "Saving accounts",
            "Checking account", "Credit amount", "Duration"]  # list of columns we will use as model inputs
target = "Risk"                                            # name of the target/output column

df_model = df[features + [target]].copy()  # create a new DataFrame df_model with only the chosen features + target
                                           # .copy() ensures we don't accidentally modify the original `df`

# ---------------------------------
# 3️⃣ Encode Categorical Columns
# ---------------------------------
cat_cols = df_model.select_dtypes(include="object").columns.drop("Risk")
# Select columns with dtype 'object' (usually strings / categorical),
# then drop "Risk" from that list because we will encode the target separately.
# NOTE: this assumes 'Risk' is among the object-dtype columns; if it's numeric, .drop("Risk") would fail.

le_dict = {}   # dictionary to keep LabelEncoder instances in memory (optional, we also save to files)

for col in cat_cols:
    le = LabelEncoder()                                    # create a new LabelEncoder for this column
    df_model[col] = le.fit_transform(df_model[col].astype(str))
    # .astype(str) ensures no errors if there are non-string types or NaNs;
    # fit_transform learns label mapping and transforms the column to numeric codes.
    le_dict[col] = le                                      # keep the encoder in the dictionary for quick use
    joblib.dump(le, f"{col}_encoder.pkl")                  # save encoder to disk for later use (e.g., in the app)

# Encode target separately
le_target = LabelEncoder()                                 # initialize encoder for the target/labels
df_model[target] = le_target.fit_transform(df_model[target])  # transform target column into numeric labels
joblib.dump(le_target, "target_encoder.pkl")               # save the target encoder so we can decode predictions later

print("✅ Encoders saved successfully!")                    # confirmation message

# ---------------------------------
# 4️⃣ Train-Test Split
# ---------------------------------
X = df_model.drop(target, axis=1)   # features matrix (all cols except the target)
y = df_model[target]                # target vector

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=1)
# Split data into training and testing sets:
# - test_size=0.2 => 20% data for testing, 80% for training
# - stratify=y => keep the same class distribution in train and test (important for imbalanced classes)
# - random_state=1 => reproducible split between runs

# ---------------------------------
# 5️⃣ Train Extra Trees Model
# ---------------------------------
et = ExtraTreesClassifier(random_state=1, class_weight="balanced", n_jobs=-1)
# Initialize ExtraTreesClassifier:
# - random_state=1 for reproducibility
# - class_weight="balanced" to handle class imbalance by weighting classes inversely to frequency
# - n_jobs=-1 uses all CPU cores to speed up training

et.fit(X_train, y_train)  # train the model on the training data

# Evaluate
y_pred = et.predict(X_test)               # predict labels for the test set
acc = accuracy_score(y_test, y_pred)      # compute accuracy: fraction of correct predictions

print(f"✅ Model trained successfully! Accuracy: {acc:.2f}")  # print accuracy (rounded to 2 decimals)

# Save model
joblib.dump(et, "extra_trees_credit_model.pkl")  # persist the trained model to disk for later use in the app
print("✅ Model saved as 'extra_trees_credit_model.pkl'")  # confirmation
