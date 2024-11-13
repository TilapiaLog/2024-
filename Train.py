import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import numpy as np

# 1. 数据加载
print("正在加载数据...")
train_data = pd.read_csv('deleted_train_data.csv')
train_target = pd.read_csv('filled_train_target.csv')
test_data = pd.read_csv('test_data.csv')
print("数据加载完成。")

# 2. 数据预处理
print("正在进行数据预处理...")
train = pd.concat([train_data, train_target], axis=1)

# 处理缺失值（示例：填充均值）
train.fillna(train.mean(), inplace=True)
test_data.fillna(test_data.mean(), inplace=True)

# 对分类变量进行独热编码
train = pd.get_dummies(train)
test_data = pd.get_dummies(test_data)

# 确保测试集和训练集具有相同的列
test_data = test_data.reindex(columns=train.columns[:-1], fill_value=0)
print("数据预处理完成。")

# 3. 特征和目标分离
X = train.drop('y', axis=1)
y = train['y']
print("特征和目标分离完成。")

# 4. 数据标准化
print("正在进行数据标准化...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_data_scaled = scaler.transform(test_data)
print("数据标准化完成。")

# 5. 模型训练
print("正在训练模型...")
model = XGBClassifier(scale_pos_weight=len(y) / sum(y), use_label_encoder=False, eval_metric='logloss')
model.fit(X_scaled, y)
print("模型训练完成。")

# 6. 预测
print("正在进行预测...")
y_pred = model.predict_proba(test_data_scaled)[:, 1]
print("预测完成。")

# 7. 结果保存
print("正在保存结果...")
result = pd.DataFrame({'idx': np.arange(1, len(y_pred) + 1), 'y_pred': y_pred})
result.to_csv('result.csv', index=False)
print("结果保存完成，文件名为 result.csv。")