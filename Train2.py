import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# 加载训练集特征数据
print("正在加载训练集特征数据...")
train_data = pd.read_csv('train_data.csv')

# 加载训练集目标变量数据
print("正在加载训练集目标变量数据...")
train_target = pd.read_csv('train_target.csv')

# 将特征和目标变量合并到一个DataFrame中，确保idx列被保留
print("正在合并训练集特征和目标变量...")
train_df = pd.merge(train_data, train_target, on='idx')

# 提取特征和目标变量，确保不删除idx列
print("正在提取特征和目标变量...")
X_train = train_df.drop(['date', 'y'], axis=1)  # 特征
y_train = train_df['y']  # 目标变量

# 加载测试集特征数据
print("正在加载测试集特征数据...")
test_data = pd.read_csv('test_data.csv')

# 初始化随机森林模型
print("正在初始化随机森林模型...")
rf = RandomForestRegressor(n_estimators=1314, random_state=66)

# 训练模型
print("正在训练模型...")
rf.fit(X_train, y_train)

# 进行预测
print("正在进行预测...")
y_pred = rf.predict(test_data)

# 创建包含idx和y_pred的DataFrame
# 确保test_data中包含idx列
result_df = pd.DataFrame({'idx': test_data['idx'], 'y_pred': y_pred})

# 将结果写入result.csv文件
print("正在将结果写入result.csv文件...")
result_df.to_csv('result.csv', index=False)

print('预测结果已保存到result.csv文件中。')
