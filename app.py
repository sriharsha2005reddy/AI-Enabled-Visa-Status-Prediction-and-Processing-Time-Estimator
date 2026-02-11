from flask import Flask, render_template, request, jsonify
import sqlite3
import joblib
from datetime import datetime

app = Flask(__name__)

# ======================
# LOAD MODELS
# ======================
time_model = joblib.load("model.pkl")          # Linear Regression → expects 14 features
status_model = joblib.load("status_model.pkl") # RandomForest → expects 13 features


# ======================
# DATABASE INITIALIZATION
# ======================
def init_db():
    conn = sqlite3.connect("visa_app.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            continent TEXT,
            education TEXT,
            experience TEXT,
            training TEXT,
            employees INTEGER,
            estab_year INTEGER,
            region TEXT,
            wage REAL,
            unit TEXT,
            fulltime TEXT,
            company_age INTEGER,
            wage_category INTEGER,
            fast_processing TEXT,
            visa_status TEXT,
            processing_time REAL
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ======================
# MAPPINGS
# ======================
continent_map = {
    "Asia": 0, "Europe": 1, "North America": 2,
    "South America": 3, "Africa": 4, "Oceania": 5
}

education_map = {
    "High School": 0, "Associate Degree": 1,
    "Bachelor's Degree": 2, "Master's Degree": 3,
    "PhD": 4
}

exp_map = {"No Experience": 0, "Has Experience": 1}
training_map = {"No": 0, "Yes": 1}

region_map = {
    "Northeast": 0, "Midwest": 1, "South": 2,
    "West": 3, "Pacific": 4, "Other": 5
}

unit_map = {"Hourly": 0, "Weekly": 1, "Monthly": 2, "Yearly": 3}
fulltime_map = {"Part-Time": 0, "Full-Time": 1}

fast_map = {"No": 0, "Yes": 1}


# ======================
# HOME PAGE
# ======================
@app.route("/")
def landing():
    return render_template("home.html")

@app.route("/form")
def form():
    return render_template("index.html")


# ======================
# PREDICTION ROUTE
# ======================
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # GET INPUT VALUES
    continent = continent_map[data["continent"]]
    education = education_map[data["education_of_employee"]]
    exp = exp_map[data["has_job_experience"]]
    training = training_map[data["requires_job_training"]]
    employees = int(data["no_of_employees"])
    estab_year = int(data["yr_of_estab"])
    region = region_map[data["region_of_employment"]]
    wage = float(data["prevailing_wage"])
    unit = unit_map[data["unit_of_wage"]]
    fulltime = fulltime_map[data["full_time_position"]]
    fast = fast_map[data["fast_processing"]]

    # AUTO FEATURES
    company_age = datetime.now().year - estab_year

    wage_category = (
        0 if wage < 40000 else
        1 if wage < 70000 else
        2 if wage < 120000 else
        3
    )

    # ======================
    # MODEL INPUTS
    # ======================

    # 13 feature input for status model
    X_status = [[
        continent, education, exp, training,
        employees, estab_year, region, wage,
        unit, fulltime, company_age,
        wage_category, fast
    ]]

    # 14 feature input for time model → requires dummy case_status
    X_time = [[
        continent, education, exp, training,
        employees, estab_year, region, wage,
        unit, fulltime,
        0,  # dummy case_status
        company_age, wage_category, fast
    ]]

    # ======================
    # PREDICTION
    # ======================
    predicted_time = round(float(time_model.predict(X_time)[0]), 2)
    status_pred = int(status_model.predict(X_status)[0])

    status_text = "Approved" if status_pred == 1 else "Denied"

    # ======================
    # REASON GENERATOR
    # ======================
    reasons = []

    if status_pred == 1:
        if exp == 1: reasons.append("Your job experience increased approval probability.")
        if wage > 70000: reasons.append("High offered wage suggests a strong job role.")
        if company_age > 10: reasons.append("Old, stable company improved approval chances.")
        if fast == 1: reasons.append("Fast processing request positively impacted approval.")
        if not reasons: reasons.append("Your profile matched approval patterns in dataset.")
    else:
        if exp == 0: reasons.append("Lack of job experience reduced approval chance.")
        if wage < 50000: reasons.append("Low wage negatively influenced approval.")
        if training == 1: reasons.append("Training requirement decreases approval rate.")
        if company_age < 3: reasons.append("Newer companies have lower approval probability.")
        if not reasons: reasons.append("Your profile matched denial patterns in dataset.")

    why_text = " • ".join(reasons)

    # ======================
    # SAVE TO DB
    # ======================
    conn = sqlite3.connect("visa_app.db")
    c = conn.cursor()
    c.execute("""
        INSERT INTO history (
            timestamp, continent, education, experience, training,
            employees, estab_year, region, wage, unit, fulltime,
            company_age, wage_category, fast_processing,
            visa_status, processing_time
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        data["continent"], data["education_of_employee"],
        data["has_job_experience"], data["requires_job_training"],
        employees, estab_year, data["region_of_employment"],
        wage, data["unit_of_wage"], data["full_time_position"],
        company_age, wage_category, data["fast_processing"],
        status_text, predicted_time
    ))

    conn.commit()
    conn.close()

    return jsonify({
        "status": status_text,
        "processing_time": predicted_time,
        "why": why_text
    })


# ======================
# HISTORY PAGE
# ======================
@app.route("/history")
def history():
    conn = sqlite3.connect("visa_app.db")
    c = conn.cursor()
    c.execute("SELECT * FROM history ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    return render_template("history.html", records=rows)


# ======================
# START APP
# ======================
if __name__ == "__main__":
    app.run(debug=True)
