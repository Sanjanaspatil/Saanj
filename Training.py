import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from pytorch_tabnet.tab_model import TabNetClassifier

# Load dataset
df = pd.read_csv("Mental Health Dataset.csv")
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df = df.loc[:, df.apply(pd.Series.nunique) > 1]
df = df.head(20000)

# --- Inject all possible categorical values (prevent unseen label error) ---
supplement = pd.DataFrame([
    {
        'Gender': 'Male',
        'Country': 'Other',
        'Occupation': 'Other',
        'self_employed': 'Yes',
        'family_history': 'Yes',
        'treatment': 'Yes',
        'Days_Indoors': 'More than 30 days',
        'Growing_Stress': 'Yes',
        'Changes_Habits': 'Yes',
        'Mental_Health_History': 'Yes',
        'Mood_Swings': 'High',
        'Coping_Struggles': 'Yes',
        'Work_Interest': 'Yes',
        'Social_Weakness': 'Yes',
        'mental_health_interview': 'Maybe',
        'care_options': 'Yes',
        'Timestamp': 'dummy'
    },
    {
        'Gender': 'Other',
        'Country': 'TestCountry',
        'Occupation': 'TestJob',
        'self_employed': 'No',
        'family_history': 'No',
        'treatment': 'No',
        'Days_Indoors': '15-30 days',
        'Growing_Stress': 'No',
        'Changes_Habits': 'No',
        'Mental_Health_History': 'No',
        'Mood_Swings': 'Low',
        'Coping_Struggles': 'No',
        'Work_Interest': 'No',
        'Social_Weakness': 'No',
        'mental_health_interview': 'No',
        'care_options': 'Not sure',
        'Timestamp': 'dummy'
    }
])
df = pd.concat([df, supplement], ignore_index=True)

# Encode categorical features
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Save label encoders
os.makedirs("models", exist_ok=True)
joblib.dump(label_encoders, "models/label_encoders.pkl")

# Feature-target split
X = df.drop(columns=["treatment", "Timestamp"])
y = df["treatment"]

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "models/scaler.pkl")

# Balance dataset
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_scaled, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
joblib.dump(X_test, 'models/X_test.pkl')
joblib.dump(y_test, 'models/y_test.pkl')

# --- TabNet ---
tabnet = TabNetClassifier(verbose=0, device_name='auto')
tabnet.fit(X_train=X_train, y_train=y_train, eval_set=[(X_test, y_test)], patience=20)
tabnet.save_model("models/tabnet")  # saves to models/tabnet.zip

# --- LightGBM ---
lgb = LGBMClassifier(n_estimators=300, learning_rate=0.03, max_depth=6, subsample=0.9, colsample_bytree=0.9, random_state=42)
lgb.fit(X_train, y_train)
joblib.dump(lgb, "models/lgb_model.pkl")

# --- Meta-model (Logistic Regression) ---
tabnet_probs = tabnet.predict_proba(X_test)[:, 1]
lgb_probs = lgb.predict_proba(X_test)[:, 1]
stacked_features = np.column_stack((tabnet_probs, lgb_probs))

meta_model = LogisticRegression()
meta_model.fit(stacked_features, y_test)
joblib.dump(meta_model, "models/meta_model.pkl")

print("\n✅ All models and encoders saved to /models")
