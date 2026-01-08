
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("Visa_Predection_Dataset.csv")

print("Initial Dataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

df['no_of_employees'] = df['no_of_employees'].fillna(
    df['no_of_employees'].median()
)
print(df['no_of_employees'].head())



df['prevailing_wage'] = df['prevailing_wage'].fillna(
    df['prevailing_wage'].median()
)
print(df['prevailing_wage'].head())

df['continent'] = df['continent'].fillna('Unknown')
df['education_of_employee'] = df['education_of_employee'].fillna('Unknown')
df['has_job_experience'] = df['has_job_experience'].fillna('Unknown')
df['requires_job_training'] = df['requires_job_training'].fillna('Unknown')
df['region_of_employment'] = df['region_of_employment'].fillna('Unknown')
df['unit_of_wage'] = df['unit_of_wage'].fillna('Unknown')
df['full_time_position'] = df['full_time_position'].fillna('Unknown')
df['case_status'] = df['case_status'].fillna('Unknown')


df = df.drop(columns=['case_id'])


df['case_status'] = df['case_status'].map({
    'Certified': 1,
    'Denied': 0
})


le = LabelEncoder()

df['continent'] = le.fit_transform(df['continent'])
df['education_of_employee'] = le.fit_transform(df['education_of_employee'])
df['has_job_experience'] = le.fit_transform(df['has_job_experience'])
df['requires_job_training'] = le.fit_transform(df['requires_job_training'])
df['region_of_employment'] = le.fit_transform(df['region_of_employment'])
df['unit_of_wage'] = le.fit_transform(df['unit_of_wage'])
df['full_time_position'] = le.fit_transform(df['full_time_position'])


np.random.seed(42)
df['processing_time'] = np.random.randint(30, 180, size=len(df))

print("\nFinal Dataset Info:")
print(df.info())

print("\nFinal Missing Values:")
print(df.isnull().sum())

df.to_csv("clean_visa_dataset.csv", index=False)


