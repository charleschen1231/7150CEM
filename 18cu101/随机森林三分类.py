import pandas as pd
import jieba
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

'''
最佳参数： {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
最佳得分： 0.6798611111111111
准确率： 0.6527777777777778
F1值： 0.7870528109028961

'''
# 加载数据集
'''
F:\master_degree\7150CEM\douban_dataset\dataset-2movies.xls  尝试加大数据量  3600条 到了0.83 
然后尝试5400条 

最佳参数： {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
最佳得分： 0.8942129629629629
准确率： 0.8824074074074074
F1值： 0.9371598218703613

看来还要提高数据量 到7200条。

最佳参数： {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
最佳得分： 0.9203125
准确率： 0.9104166666666667
F1值： 0.9530396796505278

三分类后：

最佳参数： {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 100}
最佳得分： 0.8402777777777777
准确率： 0.8347222222222223
F1 Macro： 0.3590802930121946
F1 Micro： 0.8347222222222221
F1 Weighted： 0.7687539469901276


'''
# data = pd.read_excel(r'F:\master_degree\7150CEM\douban_dataset\dataset.xls')

data = pd.read_excel(r'F:\master_degree\7150CEM\douban_dataset\dataset-4movies.xls')

# 将评分映射为情感标签
data['sentiment'] = data['rating'].apply(lambda x: 1 if x > 3 else (0 if x == 3 else -1))

# 准备特征和标签
X = data['comment']
y = data['sentiment']

# 数据预处理步骤
# 填充缺失值
imputer = SimpleImputer(strategy='most_frequent')
X = imputer.fit_transform(X.values.reshape(-1, 1)).flatten()

# 移除unicode字符
X = [re.sub(r'[^\x00-\x7F]+', '', x) for x in X]

# 移除数字
X = [re.sub(r'\d+', '', x) for x in X]

# 移除特殊字符
X = [re.sub(r'[^a-zA-Z0-9\s]', '', x) for x in X]

# 移除标点符号
X = [re.sub(r'[^\w\s]', '', x) for x in X]

# 移除多余的空格
X = [re.sub(r'\s+', ' ', x).strip() for x in X]

# 分词并移除停用词
stopwords = ['我', '你', '他',...]  # 自定义需要移除的词语列表
X = [' '.join([word for word in jieba.cut(x) if word not in stopwords]) for x in X]

# 特征工程步骤
# 统计特征
tfidf_vectorizer = TfidfVectorizer()
features = tfidf_vectorizer.fit_transform(X)

# 缩放特征
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features.toarray())

# 特征选择
selector = SelectPercentile(score_func=chi2, percentile=20)
selected_features = selector.fit_transform(np.abs(scaled_features), y)  # 使用绝对值处理非负性

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(selected_features, y, test_size=0.2, random_state=42)

# 构建随机森林分类器模型
rf_classifier = RandomForestClassifier()

# 定义要调节的参数和候选值
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 使用网格搜索进行调参优化
grid_search = GridSearchCV(rf_classifier, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# 提取参数和得分
params = grid_search.cv_results_['params']
mean_scores = grid_search.cv_results_['mean_test_score']

# 绘制曲线图
plt.figure(figsize=(12, 6))
plt.title('Random Forest Hyperparameter Tuning')
plt.xlabel('Parameter Combination')
plt.ylabel('Mean Test Score')
plt.xticks(range(len(params)), [str(param) for param in params], rotation='vertical')
plt.plot(range(len(params)), mean_scores, 'bo-', label='Mean Test Score')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# 输出最佳参数和得分
print("最佳参数：", grid_search.best_params_)
print("最佳得分：", grid_search.best_score_)

# 使用最佳参数构建随机森林模型
best_rf_classifier = RandomForestClassifier(**grid_search.best_params_)
best_rf_classifier.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = best_rf_classifier.predict(X_test)

# 计算准确率和F1值
accuracy = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_micro = f1_score(y_test, y_pred, average='micro')
f1_weighted = f1_score(y_test, y_pred, average='weighted')

print("准确率：", accuracy)
print("F1 Macro：", f1_macro)
print("F1 Micro：", f1_micro)
print("F1 Weighted：", f1_weighted)

# 绘制准确率和F1值的曲线图
param_names = [str(param) for param in params]

plt.figure(figsize=(12, 6))
plt.title('Random Forest Performance')
plt.xlabel('Parameter Combination')
plt.ylabel('Score')
plt.xticks(range(len(params)), param_names, rotation='vertical')
plt.plot(range(len(params)), [accuracy] * len(params), 'r--', label='Accuracy')
plt.plot(range(len(params)), [f1_macro] * len(params), 'g--', label='F1 Macro')
plt.plot(range(len(params)), [f1_micro] * len(params), 'b--', label='F1 Micro')
plt.plot(range(len(params)), [f1_weighted] * len(params), 'y--', label='F1 Weighted')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
