import pip

# List of libraries to install
libraries = ['numpy', 'pandas', 'matplotlib', 'scikit-learn', 'scipy', 'statsmodels', 'xgboost']

# Function to install libraries
def install_libraries(libs):
    for lib in libs:
        try:
            pip.main(['install', lib])
            print(f"{lib} installed successfully!")
        except Exception as e:
            print(f"Failed to install {lib}: {e}")

# Install libraries
      
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
import warnings
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from scipy.stats import f_oneway
from sklearn.tree import DecisionTreeClassifier

# Suppress warnings
warnings.filterwarnings("ignore")

# Read data
A1 = pd.read_excel("C:\\Users\\Karti\\.spyder-py3\\datasetsexcel\\case_study1.xlsx")
A2 = pd.read_excel("C:\\Users\\Karti\\.spyder-py3\\datasetsexcel\\case_study2.xlsx")

# Copy dataframes
df1 = A1.copy()
df2 = A2.copy()

# Remove nulls from df1
df1 = df1.loc[df1['Age_Oldest_TL'] != -99999]

# Remove columns with too many nulls from df2
columns_to_be_removed = [col for col in df2.columns if df2[col].eq(-99999).sum() > 10000]
df2 = df2.drop(columns_to_be_removed, axis=1)
df2 = df2.loc[:, df2.ne(-99999).all()]

# Merge dataframes
df = pd.merge(df1, df2, how='inner', on='PROSPECTID')

# Chi-square test for categorical variables
for col in ['MARITALSTATUS', 'EDUCATION', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']:
    chi2, pval, _, _ = chi2_contingency(pd.crosstab(df[col], df['Approved_Flag']))
    print(f"{col} --- p-value: {pval}")

# VIF for numerical columns
numeric_columns = df.select_dtypes(include=['number']).columns.drop(['PROSPECTID', 'Approved_Flag'])
vif_data = df[numeric_columns]
columns_to_keep = []

for i, col in enumerate(numeric_columns):
    vif_value = variance_inflation_factor(vif_data.values, i)
    if vif_value <= 6:
        columns_to_keep.append(col)

# ANOVA for selected numerical columns
columns_to_keep_numerical = []

for col in columns_to_keep:
    p_value = f_oneway(*[df[col][df['Approved_Flag'] == f'P{i}'] for i in range(1, 5)])[1]
    if p_value <= 0.05:
        columns_to_keep_numerical.append(col)

# Final features
features = columns_to_keep_numerical + ['MARITALSTATUS', 'EDUCATION', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']
df = df[features + ['Approved_Flag']]

# Label encoding for categorical features
categorical_cols = ['MARITALSTATUS', 'EDUCATION', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']

for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# Model fitting - Random Forest
y = df['Approved_Flag']
X = df.drop(['Approved_Flag'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=200, random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy:.2f}")

# Model fitting - XGBoost
xgb_classifier = xgb.XGBClassifier(objective='multi:softmax', num_class=4)
xgb_classifier.fit(X_train, y_train)
y_pred = xgb_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"XGBoost Accuracy: {accuracy:.2f}")

# Model fitting - Decision Tree
dt_model = DecisionTreeClassifier(max_depth=20, min_samples_split=10)
dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Decision Tree Accuracy: {accuracy:.2f}")
