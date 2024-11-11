import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

california = fetch_california_housing()
df = pd.DataFrame(california.data, columns=california.feature_names)
df['Price'] = california.target
print(df.columns)
n_f = df.select_dtypes(include=[np.number]).columns
c_f = df.select_dtypes(exclude=[np.number]).columns
print(f"Nu fe: {len(n_f)},res: {len(c_f)}")


sns.lineplot(x=df['A'], y=df['Price'])
plt.xlabel('No of rooms')
plt.ylabel('Poh')
plt.title('House')
plt.show()

miss = df.isnull().sum()
print(f"Mis val :\n{miss}")

imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)


X = df_imputed.drop('Price', axis=1)
y = df_imputed['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)


l= LinearRegression()
l.fit(X_train, y_train)
y_pred_l = l.predict(X_test)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm = SVR()
svm.fit(X_train_scaled, y_train)
y_pred_svm = svm.predict(X_test_scaled)


a_lr = accuracy_score(y_test, y_pred_l.round()) 
a_svm = accuracy_score(y_test, y_pred_svm.round())

p_lr = precision_score(y_test, y_pred_l.round(), average='binary', zero_division=0)
p_svm = precision_score(y_test, y_pred_svm.round(), average='binary', zero_division=0)

recall_lr = recall_score(y_test, y_pred_l.round(), average='binary', zero_division=0)
recall_svm = recall_score(y_test, y_pred_svm.round(), average='binary', zero_division=0)

f1_lr = f1_score(y_test, y_pred_l.round(), average='binary')
f1_svm = f1_score(y_test, y_pred_svm.round(), average='binary')

print(f"Linear Regression - Accuracy: {accuracy_lr}, Precision: {precision_lr}, Recall: {recall_lr}, F1: {f1_lr}")
print(f"SVM - Accuracy: {accuracy_svm}, Precision: {precision_svm}, Recall: {recall_svm}, F1: {f1_svm}")
