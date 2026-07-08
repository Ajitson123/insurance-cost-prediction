import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
v = pd.read_csv("HeartDiseaseTrain-Test.csv")
print(v['target'].value_counts())
print(v.describe())
sns.countplot(x='target', data= v)
plt.title('Heart Disease Distribution')
sns.heatmap(v.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm', fmt='.2f')
cat_cols = v.select_dtypes(include='object').columns
v = pd.get_dummies(v, columns=cat_cols, drop_first=True)
print(v.shape)
#split 
x = v.drop('target', axis =1)
y = v['target']
x_train , x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
print(x_train.shape)
models = {
    'logistic regression' :LogisticRegression(),
    'knn' : KNeighborsClassifier(),
    'decision tree': DecisionTreeClassifier(),
    'random forest': RandomForestClassifier(),
    'adaboost': AdaBoostClassifier(),
    'gradient boosting':GradientBoostingClassifier(),
    'svm': SVC(),
    'xgboost': XGBClassifier(evel_metric='logloss')

}
result = []
for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    result.append({'model': name, 'accuracy':round(acc*100, 2)})
    print(f"{name}: {acc*100:.2f}")
print(pd.DataFrame(result).sort_values('accuracy', ascending=False))
best = RandomForestClassifier(n_estimators=100, random_state=42)
best.fit(x_train, y_train)
y_pred = best.predict(x_test)
print(classification_report(y_pred, y_test))
print('roc auc:', roc_auc_score(y_test, y_pred))
feat_imp = pd.Series(best.feature_importances_, index=v.drop('target', axis=1).columns)
feat_imp.sort_values().plot(kind='barh', figsize=(8, 10))
plt.title('feature importance- Heart Disease')
plt.show()
import joblib
joblib.dump(best, 'heart_disease_model.pkl')
print('model saved')