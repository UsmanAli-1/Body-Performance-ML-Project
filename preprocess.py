## Step 1 - Import libraries and load dataset
import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np
# Encode 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier



df = pd.read_csv("bodyPerformance.csv")

## checking dataset basic 

# print(df.shape)       # rows, columns
# print(df.head())      # first 5 rows
# print(df.info())      # column types, 3





## step 2 ===================================
## Target variable distribution

# print(df['class'].value_counts())

# df['class'].value_counts().plot(kind='bar', title="Class Distribution")
# plt.show()

# print(df['gender'].value_counts())
# df['gender'].value_counts().plot(kind='bar', title="gender")
# plt.show()





## step 3  ==============================
## Summary statistics


# print(df.describe())

## Quick histograms for numeric columns
# df.hist(figsize=(12, 10), bins=30)
# plt.tight_layout()
# plt.show()






## step 4 ================================ 
## handle outliers with IQR (winsorization)

def cap_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df[column] = np.where(df[column] < lower_bound, lower_bound,
                  np.where(df[column] > upper_bound, upper_bound, df[column]))
    return df

# print("Remaining data shape:", df.shape)






## step 5 =============================
## handle dublicate records 

## Check number of duplicates (1 dublicate row)
# print("Duplicate rows before:", df.duplicated().sum()) 


## Remove duplicates
df = df.drop_duplicates()

# print("Remaining data shape:", df.shape)






## step 6 ==============================
##  Feature Engineering (BMI)
## BMI= Weight (kg) / Height(m)sqrt(2)

# Convert height from cm to meters
df["Height_m"] = df["height_cm"] / 100

# Create BMI
df["BMI"] = df["weight_kg"] / (df["Height_m"] ** 2)

# Drop temporary column and original height/weight 
df = df.drop(columns=["height_cm", "weight_kg", "Height_m"])

# print(df.head())





## step 7 ===================================
##Encoding gender and class 

##one hot encoding 
df = pd.get_dummies(df, columns=["gender"], drop_first=True)

# print(df.head()) 

# ## Encode class as ordinal encoding
df["class"] = df["class"].map({"A": 0, "B": 1, "C": 2, "D": 3})

# print(df[["gender", "class"]].head())






## step 8 ===================================
##Feature Scaling

## Split features (X) and target (y)
X = df.drop(columns=["class"])   # all predictors
y = df["class"]                  # target

## Apply MinMaxScaler (scales everything between 0 and 1)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

## Convert back to DataFrame for readability
X = pd.DataFrame(X_scaled, columns=X.columns)

# print("After scaling:")
# print(X.head())



# # Combine scaled features with target for export
# df_scaled = pd.concat([X, y], axis=1)


## step 9 ===================================
## Train-test split

# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# print("Training set shape:", X_train.shape)
# print("Test set shape:", X_test.shape)



##////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


## step 10 ===================================
# ========================
## Logistic Regression (baseline model) ==================1st modal 
# ========================


# # Initialize logistic regression model
# log_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)

# # Train on training data
# log_reg.fit(X_train, y_train)

# # Predict on test set
# y_pred = log_reg.predict(X_test)

# # Evaluate performance
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

# print("\nConfusion Matrix:")
# print(confusion_matrix(y_test, y_pred))


##////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# ========================
# K-Nearest Neighbors (KNN) ======================2nd modal
# ========================
# knn_clf = KNeighborsClassifier(n_neighbors=5)  # default k=5
# knn_clf.fit(X_train, y_train)

# y_pred_knn = knn_clf.predict(X_test)

# print("\nKNN Accuracy:", accuracy_score(y_test, y_pred_knn))
# print("\nClassification Report (KNN):\n", classification_report(y_test, y_pred_knn))
# print("\nConfusion Matrix (KNN):\n", confusion_matrix(y_test, y_pred_knn))


##////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# ==========================
# # Initialize Random Forest ======================= 3rd modal
# ==========================

# rf = RandomForestClassifier(
#     n_estimators=200,   # number of trees
#     max_depth=None,    # let it grow fully
#     random_state=42,
#     n_jobs=-1          # use all CPU cores for speed
# )

# # Train
# rf.fit(X_train, y_train)

# # Predict
# y_pred_rf = rf.predict(X_test)

# # Evaluate
# print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
# print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))
# print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))


##////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


# ===================================
# Step 11 - XGBoost Classifier  ========================4th modal
# ===================================

# Initialize XGBoost model
xgb = XGBClassifier(
    n_estimators=300,     # number of trees
    learning_rate=0.1,    # step size shrinkage
    max_depth=6,          # depth of trees
    random_state=42,
    n_jobs=-1,            # use all CPU cores
    objective="multi:softmax", # multi-class classification
    num_class=4           # number of classes (A,B,C,D → 0,1,2,3)
)

# Train
xgb.fit(X_train, y_train)

# Predict
y_pred_xgb = xgb.predict(X_test)

# Evaluate
# print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
# print("\nClassification Report:\n", classification_report(y_test, y_pred_xgb))
# print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))



# # ==========================================================


# Save XGBoost model (modal dump)
joblib.dump(xgb, "xgboost_model.pkl")

# Save scaler (important for preprocessing when making predictions later)
joblib.dump(scaler, "scaler.pkl")

# print("✅ XGBoost model and scaler saved successfully!")