
#MILESTONE-1
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



#MILESTONE-2
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
df.to_csv("milestone2.csv", index=False)


#MILESTONE-3
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


#loading preprocessed dataset
df = pd.read_csv("milestone2.csv")
print("Dataset Loaded Successfully\n")
print(df.head())

#select features and target
X = df.drop("processing_time", axis=1)
y = df["processing_time"]

#train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#linear regression train and evaluation
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

lr_mae = mean_absolute_error(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_r2 = r2_score(y_test, lr_pred)

print("\n===== Linear Regression Results =====")
print("MAE :", lr_mae)
print("RMSE:", lr_rmse)
print("R²  :", lr_r2)


#random forest train and evaluation
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print("\n===== Random Forest Results =====")
print("MAE :", rf_mae)
print("RMSE:", rf_rmse)
print("R²  :", rf_r2)

#gradient boosting train and evaluation
gbr = GradientBoostingRegressor(random_state=42)
gbr.fit(X_train, y_train)
gbr_pred = gbr.predict(X_test)

gbr_mae = mean_absolute_error(y_test, gbr_pred)
gbr_rmse = np.sqrt(mean_squared_error(y_test, gbr_pred))
gbr_r2 = r2_score(y_test, gbr_pred)

print("\n===== Gradient Boosting Results =====")
print("MAE :", gbr_mae)
print("RMSE:", gbr_rmse)
print("R²  :", gbr_r2)

#best model
model_errors = {
    "Linear Regression": lr_rmse,
    "Random Forest": rf_rmse,
    "Gradient Boosting": gbr_rmse
}

best_model_name = min(model_errors, key=model_errors.get)
print("\n===== Best Model Identified =====")
print("Best Model =", best_model_name)
if best_model_name == "Linear Regression":
    best_model = lr
elif best_model_name == "Random Forest":
    best_model = rf
else:
    best_model = gbr

#final predictions
final_pred = best_model.predict(X_test)

#final model evaluation
final_mae = mean_absolute_error(y_test, final_pred)
final_rmse = np.sqrt(mean_squared_error(y_test, final_pred))
final_r2 = r2_score(y_test, final_pred)

print("\n===== FINAL MODEL PERFORMANCE =====")
print("Final MAE :", final_mae)
print("Final RMSE:", final_rmse)
print("Final R²  :", final_r2)

#saving the best model
output = pd.DataFrame({
    "Actual Processing Time": y_test,
    "Predicted Processing Time": final_pred
})

output.to_csv("milestone3_predictions.csv", index=False)

print("\nMilestone 3 Completed Successfully!")
print("Prediction file saved as milestone3_predictions.csv")