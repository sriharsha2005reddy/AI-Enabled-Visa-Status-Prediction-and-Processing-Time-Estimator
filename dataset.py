
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
#loading
df = pd.read_csv("Visa_Predection_Dataset.csv")

print("Initial Dataset Info:")
print(df.info())
#missing
print("\nMissing Values:")
print(df.isnull().sum())

#handling
#numerical
df['no_of_employees'] = df['no_of_employees'].fillna(
    df['no_of_employees'].median()
)
print(df['no_of_employees'].head())


df['prevailing_wage'] = df['prevailing_wage'].fillna(
    df['prevailing_wage'].median()
)
print(df['prevailing_wage'].head())

#categorical
df['continent'] = df['continent'].fillna('Unknown')
df['education_of_employee'] = df['education_of_employee'].fillna('Unknown')
df['has_job_experience'] = df['has_job_experience'].fillna('Unknown')
df['requires_job_training'] = df['requires_job_training'].fillna('Unknown')
df['region_of_employment'] = df['region_of_employment'].fillna('Unknown')
df['unit_of_wage'] = df['unit_of_wage'].fillna('Unknown')
df['full_time_position'] = df['full_time_position'].fillna('Unknown')
df['case_status'] = df['case_status'].fillna('Unknown')


df = df.drop(columns=['case_id'])

#target
df['case_status'] = df['case_status'].map({
    'Certified': 1,
    'Denied': 0
})

#encoding categorical
le = LabelEncoder()

df['continent'] = le.fit_transform(df['continent'])
df['education_of_employee'] = le.fit_transform(df['education_of_employee'])
df['has_job_experience'] = le.fit_transform(df['has_job_experience'])
df['requires_job_training'] = le.fit_transform(df['requires_job_training'])
df['region_of_employment'] = le.fit_transform(df['region_of_employment'])
df['unit_of_wage'] = le.fit_transform(df['unit_of_wage'])
df['full_time_position'] = le.fit_transform(df['full_time_position'])

#processing in days
np.random.seed(42)
df['processing_time'] = np.random.randint(30, 180, size=len(df))

print("\nFinal Dataset Info:")
print(df.info())

print("\nFinal Missing Values:")
print(df.isnull().sum())

df.to_csv("clean_visa_dataset.csv", index=False)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#loading preprocessed dataset
df = pd.read_csv("clean_visa_dataset.csv")

print("Dataset loaded successfully")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

#visualization
plt.figure(figsize=(6,4))
sns.countplot(x='case_status', data=df)
plt.title("Visa Status Distribution")
plt.xlabel("Case Status")
plt.ylabel("Count")
plt.show()


plt.figure(figsize=(7,4))
plt.hist(df['prevailing_wage'], bins=30)
plt.title("Prevailing Wage Distribution")
plt.xlabel("Wage")
plt.ylabel("Count")
plt.show()
#correlation
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

#feature engineering
df['company_age'] = 2025 - df['yr_of_estab']


df['wage_category'] = pd.cut(
    df['prevailing_wage'],
    bins=[0, 50000, 100000, 200000, df['prevailing_wage'].max()],
    labels=[0, 1, 2, 3]
)

df['fast_processing'] = df['processing_time'].apply(
    lambda x: 1 if x <= 90 else 0
)

print("\nDataset after Feature Engineering:")
print(df.info())


