import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
v = pd.read_csv("insurance.csv")
v['sex'] = v['sex'].map({'male':1, 'female':0})
v['smoker'] = v['smoker'].map({'yes':1, 'no':0})
v = pd.get_dummies(v, columns=['region'], drop_first=True)
x = v.drop('charges', axis=1)
y = v['charges']
x_train , x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
print("ready:", x_train.shape)
models = {
    'linear regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso' : Lasso(alpha=1.0),
    'Random forest' : RandomForestRegressor(n_estimators=100, random_state=42),
    'gradient boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}
results = []
for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append({''
        'Model': name,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2 Score': r2
    })
    print(f"{name} - MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2 Score: {r2:.2f}")
results_df = pd.DataFrame(results)
print(results_df)
best_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
best_model.fit(x_train, y_train)
feature_nammes = v.drop('charges', axis=1).columns
feature_importances = best_model.feature_importances_
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=feature_nammes)
plt.title("Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()
import joblib
joblib.dump(best_model, 'insurance_cost_model.pkl')
joblib.dump(sc, 'scaler.pkl')
print("model saved")
sample = sc.transform([[35, 1, 28.5, 2, 1, 0, 0, 1]])
predicted = best_model.predict(sample)
print(f"predicted insurance cost: {predicted[0]:,.2f}") 

