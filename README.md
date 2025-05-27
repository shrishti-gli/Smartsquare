# Smartsquare
#  STEP 1: Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

#  STEP 2: Load Dataset
# Upload dataset through Colab or use a sample dataset
from google.colab import files
uploaded = files.upload()  # Upload 'house_prices.csv'

df = pd.read_csv(next(iter(uploaded)))
print("Data Loaded Successfully\n")

# STEP 3: Initial Overview
print("Shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nData Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

#  STEP 4: Handling Missing Values
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print("\nMissing Values:\n", missing)

# Impute numerical with median, categorical with mode
num_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(include=['object']).columns

imputer_num = SimpleImputer(strategy='median')
imputer_cat = SimpleImputer(strategy='most_frequent')

df[num_cols] = imputer_num.fit_transform(df[num_cols])
df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])

print("\n Missing Values Handled")

# STEP 5: Feature Selection and Engineering
# Encode categorical variables
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Correlation matrix
corr_matrix = df.corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
plt.title("Correlation Matrix")
plt.show()

# Select top features highly correlated with target
target = 'SalePrice'
cor_target = abs(corr_matrix[target])
selected_features = cor_target[cor_target > 0.5].index.tolist()
print("\n Selected Features:\n", selected_features)

#  STEP 6: Ensure Data Consistency
# Check for duplicate rows
duplicates = df.duplicated().sum()
print(f"\n Duplicates Found: {duplicates}")
if duplicates > 0:
    df = df.drop_duplicates()
    print(" Duplicates Removed")

# Check for negative values in normally positive columns
for col in selected_features:
    if (df[col] < 0).sum() > 0:
        print(f"⚠ Negative values found in {col}")

#  STEP 7: Summary Statistics and Insights
summary_stats = df[selected_features].describe().T
summary_stats['skewness'] = df[selected_features].skew()
print("\n Summary Statistics:\n", summary_stats)

# STEP 8: Identify Outliers
plt.figure(figsize=(10, 5))
sns.boxplot(data=df[selected_features].drop(target, axis=1))
plt.xticks(rotation=45)
plt.title("Boxplots for Selected Features")
plt.show()

# Outlier handling (optional): clip extreme values
for col in selected_features:
    q_low = df[col].quantile(0.01)
    q_high = df[col].quantile(0.99)
    df[col] = df[col].clip(q_low, q_high)

print("\n Outliers Clipped (1st and 99th percentile)")

#  STEP 9: Data Transformation
scaler = StandardScaler()
df[selected_features] = scaler.fit_transform(df[selected_features])

print("\n Data Scaled with StandardScaler")

# STEP 10: Initial Visual Representations
# SalePrice Distribution
plt.figure(figsize=(6, 4))
sns.histplot(df[target], bins=30, kde=True)
plt.title("Distribution of SalePrice")
plt.show()

# Heatmap of Selected Features
plt.figure(figsize=(10, 8))
sns.heatmap(df[selected_features].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation of Selected Features")
plt.show()

# Relationship between OverallQual and SalePrice
plt.figure(figsize=(6, 4))
sns.scatterplot(x=df['OverallQual'], y=df['SalePrice'])
plt.title("OverallQual vs SalePrice")
plt.xlabel("Overall Quality")
plt.ylabel("Sale Price")
plt.show()

print("\n Initial Visual Representation Complete")

#  DATA IS NOW READY FOR MODELING
