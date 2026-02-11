import sqlite3

conn = sqlite3.connect("visa_app.db")
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    continent INTEGER,
    education INTEGER,
    has_job_experience INTEGER,
    requires_job_training INTEGER,
    no_of_employees INTEGER,
    yr_of_estab INTEGER,
    region_of_employment INTEGER,
    prevailing_wage REAL,
    unit_of_wage INTEGER,
    full_time_position INTEGER,
    company_age INTEGER,
    wage_category INTEGER,
    fast_processing INTEGER,
    visa_status TEXT,
    predicted_processing_time REAL,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()
conn.close()

print("✔ Database created successfully!")
