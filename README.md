import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

# Load and preprocess dataset
df = pd.read_csv("Mental Health Dataset.csv")
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df = df.loc[:, df.apply(pd.Series.nunique) > 1]
df = df.head(20000)

# Encode categorical columns
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define features and target
X = df.drop("treatment", axis=1)
y = df["treatment"]

# Scale and balance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_scaled, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# --- TabNet Transfer Learning ---
tabnet = TabNetClassifier(verbose=0, device_name='auto')
tabnet.fit(X_train=X_train, y_train=y_train, eval_set=[(X_test, y_test)], patience=20)

tabnet_probs = tabnet.predict_proba(X_test)[:, 1]
tabnet_preds = (tabnet_probs > 0.5).astype(int)

# --- LightGBM ---
lgb = LGBMClassifier(n_estimators=300, learning_rate=0.03, max_depth=6, subsample=0.9, colsample_bytree=0.9, random_state=42)
lgb.fit(X_train, y_train)
lgb_probs = lgb.predict_proba(X_test)[:, 1]
lgb_preds = lgb.predict(X_test)

# --- Meta-model  ---
meta_model = LogisticRegression()
stacked_features = np.column_stack((tabnet_probs, lgb_probs))
meta_model.fit(stacked_features, y_test)
hybrid_probs = meta_model.predict_proba(stacked_features)[:, 1]
hybrid_preds = meta_model.predict(stacked_features)


# --- Evaluation Function ---
def evaluate_model(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    mis = np.sum(cm) - np.trace(cm)
    print(f"\n✅ {name} Accuracy: {acc:.4f}")
    print(f"🔍 {name} Misclassifications: {mis}")
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap='Blues')
    plt.title(f"{name} Confusion Matrix")
    plt.show()

evaluate_model("TabNet", y_test, tabnet_preds)
evaluate_model("LightGBM", y_test, lgb_preds)
evaluate_model("Hybrid Model", y_test, hybrid_preds)

# --- Export Predictions & Risk Assessment ---
def assign_risk(prob):
    if prob >= 0.8:
        return "High"
    elif prob >= 0.5:
        return "Medium"
    else:
        return "Low"

export_df = pd.DataFrame({
    "Actual": y_test.values,
    "TabNet_Prob": tabnet_probs,
    "LightGBM_Prob": lgb_probs,
    "Hybrid_Prob": hybrid_probs,
    "Hybrid_Prediction": hybrid_preds,
    "Risk_Level": [assign_risk(p) for p in hybrid_probs]
})

export_df.to_csv("hybrid_predictions_with_risk.csv", index=False)
print("\n📤 Predictions with risk levels exported to 'hybrid_predictions_with_risk.csv'.")

# --- Cross Validation ---
print("\n🔄 Running Stratified K-Fold Cross Validation on Hybrid Stack...")
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = []
for train_idx, test_idx in kf.split(X_res, y_res):
    X_tr, X_te = X_res[train_idx], X_res[test_idx]
    y_tr, y_te = y_res[train_idx], y_res[test_idx]

    tabnet = TabNetClassifier(verbose=0)
    tabnet.fit(X_train=X_tr, y_train=y_tr, eval_set=[(X_te, y_te)], patience=10)

    lgb = LGBMClassifier(n_estimators=200)
    lgb.fit(X_tr, y_tr)

    tab_probs = tabnet.predict_proba(X_te)[:, 1]
    lgb_probs = lgb.predict_proba(X_te)[:, 1]
    stacked = np.column_stack((tab_probs, lgb_probs))

    meta = LogisticRegression()
    meta.fit(stacked, y_te)
    final_preds = meta.predict(stacked)

    acc = accuracy_score(y_te, final_preds)
    cv_scores.append(acc)

print(f"✅ Cross-Validation Accuracy Scores: {cv_scores}")
print(f"📊 Mean CV Accuracy: {np.mean(cv_scores):.4f}")
