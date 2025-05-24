from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import string
import random
from flask_sqlalchemy import SQLAlchemy
from lightgbm import LGBMClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

app = Flask(__name__)

# --- Database config ---
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///mental_health.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- Database model ---
class Submission(db.Model):
    id = db.Column(db.String(12), primary_key=True)
    name = db.Column(db.String(100))
    data = db.Column(db.PickleType)
    tabnet_score = db.Column(db.Float)
    lgb_score = db.Column(db.Float)
    meta_score = db.Column(db.Float)
    treatment_needed = db.Column(db.String(10))
    risk_level = db.Column(db.String(10))
    probable_condition = db.Column(db.String(100))

def generate_id(size=12):
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choice(chars) for _ in range(size))

@app.before_request
def create_tables():
    db.create_all()

# --- Load models and tools ---
label_encoders = joblib.load('models/label_encoders.pkl')
scaler = joblib.load('models/scaler.pkl')
tabnet = TabNetClassifier()
tabnet.load_model('models/tabnet.zip')
lgb_model = joblib.load('models/lgb_model.pkl')
meta_model = joblib.load('models/meta_model.pkl')
X_test = joblib.load('models/X_test.pkl')
y_test = joblib.load('models/y_test.pkl')


def infer_disorder(row):
    # Convert string inputs to numerical scale if needed
    mood_swing_map = {'Low': 1, 'Medium': 2, 'High': 3}
    yes_no_map = {'No': 1, 'Yes': 3}

    ms = mood_swing_map.get(row.get('Mood_Swings'), 0)
    cs = yes_no_map.get(row.get('Coping_Struggles'), 0)
    gs = yes_no_map.get(row.get('Growing_Stress'), 0)
    ch = yes_no_map.get(row.get('Changes_Habits'), 0)

    if ms > 2 and cs > 2:
        return 'Bipolar Disorder'
    elif gs > 2 and ch > 2:
        return 'Depression'
    elif ms <= 2 and cs <= 2 and gs <= 2:
        return 'Anxiety'
    else:
        return 'General'


def assign_risk(prob):
    if prob >= 0.8:
        return "High"
    elif prob >= 0.5:
        return "Medium"
    else:
        return "Low"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('form.html')

    # POST method handling below
    input_data = {key: request.form[key] for key in request.form}
    user_name = input_data.pop('name', None)  # Remove and store name separately
    disorder_type = infer_disorder(input_data)

    df = pd.DataFrame([input_data])

    for col in df.columns:
        if col in label_encoders:
            df[col] = label_encoders[col].transform(df[col])

    X_scaled = scaler.transform(df)

    tabnet_prob = tabnet.predict_proba(X_scaled)[:, 1]
    lgb_prob = lgb_model.predict_proba(X_scaled)[:, 1]
    stacked_features = np.column_stack((tabnet_prob, lgb_prob))
    final_prob = meta_model.predict_proba(stacked_features)[:, 1]
    prediction = (final_prob > 0.5).astype(int)[0]

    risk = assign_risk(final_prob[0])
    treatment_needed = "Yes" if prediction == 1 else "No"

    # Save to DB
    submission_id = generate_id()
    submission = Submission(
        id=submission_id,
        name=user_name,
        data=input_data,
        tabnet_score=float(tabnet_prob[0]),
        lgb_score=float(lgb_prob[0]),
        meta_score=float(final_prob[0]),
        treatment_needed=treatment_needed,
        risk_level=risk,
        probable_condition=disorder_type
    )
    db.session.add(submission)
    db.session.commit()

    # Generate classification report
    tabnet_probs_test = tabnet.predict_proba(X_test)[:, 1]
    lgb_probs_test = lgb_model.predict_proba(X_test)[:, 1]
    stacked_features_test = np.column_stack((tabnet_probs_test, lgb_probs_test))
    final_preds_test = meta_model.predict(stacked_features_test)
    classification_report_dict = classification_report(y_test, final_preds_test, output_dict=True)

    return render_template("result.html",
                           prediction=treatment_needed,
                           risk_level=risk,
                           tabnet_score=round(float(tabnet_prob[0]), 3),
                           lgb_score=round(float(lgb_prob[0]), 3),
                           meta_score=round(float(final_prob[0]), 3),
                           classification_report=classification_report_dict,
                           submission_id=submission_id,
                           disorder_type=disorder_type)

@app.route('/retrieve', methods=['GET', 'POST'])
def retrieve():
    if request.method == 'POST':
        sub_id = request.form.get('submission_id')
        submission = Submission.query.get(sub_id)
        if submission:
            return render_template('display_submission.html', submission=submission)
        else:
            return "No submission found with this ID.", 404
    return render_template('retrieve_form.html')

if __name__ == '__main__':
    app.run(debug=True)
