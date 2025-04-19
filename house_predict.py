import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. 讀取資料
df = pd.read_csv("taiwan_house_prices.csv")

# 2. 特徵與目標欄位
X = df[['區域', '建物坪數', '屋齡']]
y = df['房價（萬元）']

# 3. 類別型資料處理（One-Hot Encoding）
encoder = OneHotEncoder(sparse=False)
X_encoded = encoder.fit_transform(X[['區域']])

# 結合數值型資料
X_final = pd.concat([
    pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(['區域'])),
    X[['建物坪數', '屋齡']].reset_index(drop=True)
], axis=1)

# 4. 切分訓練與測試資料
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# 5. 建立與訓練模型
model = LinearRegression()
model.fit(X_train, y_train)

# 6. 預測與評估
y_pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

# 7. 範例預測
sample = pd.DataFrame({
    '區域': ['台北市'],
    '建物坪數': [35.0],
    '屋齡': [7]
})

sample_encoded = encoder.transform(sample[['區域']])
sample_final = pd.concat([
    pd.DataFrame(sample_encoded, columns=encoder.get_feature_names_out(['區域'])),
    sample[['建物坪數', '屋齡']]
], axis=1)

prediction = model.predict(sample_final)
print("預測房價：", round(prediction[0], 2), "萬元")
hello