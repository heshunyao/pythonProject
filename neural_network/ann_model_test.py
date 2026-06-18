import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow import keras

def predict_new_phone(model, scaler, feature_names, phone_features):
    """
    预测新手机的价格区间
    """
    # 转换为DataFrame
    phone_df = pd.DataFrame([phone_features])

    # 确保所有特征都存在
    for feature in feature_names:
        if feature not in phone_df.columns:
            phone_df[feature] = 0

    # 重新排序列以匹配训练数据
    phone_df = phone_df[feature_names]

    # 特征缩放
    phone_scaled = scaler.transform(phone_df)

    # 预测
    prediction_prob = model.predict(phone_scaled)
    prediction = np.argmax(prediction_prob, axis=1)[0]

    price_categories = {
        0: '低端手机',
        1: '中低端手机',
        2: '中高端手机',
        3: '高端手机'
    }

    return {
        'price_range': prediction,
        'category': price_categories[prediction],
        'probabilities': prediction_prob[0].tolist(),
        'confidence': np.max(prediction_prob[0])
    }


# 示例使用

# 加载已保存的模型
model = keras.models.load_model('mobile_price_classifier.h5')

# 新手机特征示例
new_phone = {
    'battery_power': 1500,
    'blue': 1,
    'clock_speed': 2.5,
    'dual_sim': 1,
    'fc': 8,
    'four_g': 1,
    'int_memory': 32,
    'm_dep': 0.6,
    'mobile_wt': 150,
    'n_cores': 4,
    'pc': 12,
    'ram': 2000,
    'talk_time': 10,
    'three_g': 1,
    'touch_screen': 1,
    'wifi': 1
}

# 预测
result = predict_new_phone(model, scaler, feature_names, new_phone)
print(f"预测结果: {result}")
