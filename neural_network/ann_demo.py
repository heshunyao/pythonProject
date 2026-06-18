import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import warnings
import joblib

warnings.filterwarnings('ignore')

# 设置随机种子以确保可重复性
np.random.seed(42)
tf.random.set_seed(42)


# 1. 数据加载和探索
def load_and_explore_data():
    """
    加载并探索数据集
    """
    # 这里假设数据已经加载到DataFrame中
    # 如果是CSV文件，使用: df = pd.read_csv('mobile_data.csv')

    # 创建示例数据（实际使用时从文件加载）
    np.random.seed(42)
    n_samples = 2000

    data = {
        'battery_power': np.random.randint(500, 2000, n_samples),
        'blue': np.random.randint(0, 2, n_samples),
        'clock_speed': np.random.uniform(0.5, 3.0, n_samples),
        'dual_sim': np.random.randint(0, 2, n_samples),
        'fc': np.random.randint(0, 20, n_samples),
        'four_g': np.random.randint(0, 2, n_samples),
        'int_memory': np.random.randint(2, 64, n_samples),
        'm_dep': np.random.uniform(0.1, 1.0, n_samples),
        'mobile_wt': np.random.randint(80, 200, n_samples),
        'n_cores': np.random.randint(1, 9, n_samples),
        'pc': np.random.randint(0, 20, n_samples),
        'px_height': np.random.randint(0, 2000, n_samples),
        'px_width': np.random.randint(500, 2000, n_samples),
        'ram': np.random.randint(256, 4000, n_samples),
        'sc_h': np.random.randint(5, 20, n_samples),
        'sc_w': np.random.randint(3, 15, n_samples),
        'talk_time': np.random.randint(2, 20, n_samples),
        'three_g': np.random.randint(0, 2, n_samples),
        'touch_screen': np.random.randint(0, 2, n_samples),
        'wifi': np.random.randint(0, 2, n_samples),
        'price_range': np.random.randint(0, 4, n_samples)  # 目标变量
    }

    df = pd.DataFrame(data)
    print("数据集形状:", df.shape)
    print("\n前5行数据:")
    print(df.head())
    print("\n数据基本信息:")
    print(df.info())
    print("\n描述性统计:")
    print(df.describe())

    return df


# 2. 特征工程
def feature_engineering(df):
    """
    创建新特征以提升模型性能
    """
    df = df.copy()

    # 1. 像素总数（可能影响显示质量）
    df['total_pixels'] = df['px_height'] * df['px_width']

    # 2. 屏幕面积
    df['screen_area'] = df['sc_h'] * df['sc_w']

    # 3. 像素密度
    df['pixel_density'] = df['total_pixels'] / (df['screen_area'] + 1e-8)

    # 4. 电池功率与重量的比率
    df['battery_to_weight'] = df['battery_power'] / df['mobile_wt']

    # 5. 摄像头总数
    df['total_cameras'] = df['fc'] + df['pc']

    # 6. 内存与RAM比率
    df['memory_to_ram'] = df['int_memory'] / (df['ram'] / 1024 + 1e-8)  # 转换为GB

    # 7. 网络功能得分
    df['network_score'] = df['three_g'] + df['four_g'] + df['wifi'] + df['blue']

    # 8. 处理器性能指标
    df['processor_power'] = df['clock_speed'] * df['n_cores']

    # 9. 电池续航指标
    df['battery_efficiency'] = df['talk_time'] / df['battery_power'] * 1000

    # 10. 屏幕分辨率类别
    df['resolution_category'] = pd.cut(df['total_pixels'],
                                       bins=[0, 500000, 1000000, 1500000, float('inf')],
                                       labels=[0, 1, 2, 3])

    # 删除原始像素列（保留衍生特征）
    df = df.drop(['px_height', 'px_width', 'sc_h', 'sc_w'], axis=1)

    return df


# 3. 数据预处理
def preprocess_data(df):
    """
    数据预处理和划分
    """
    # 分离特征和目标
    X = df.drop('price_range', axis=1)
    y = df['price_range']
    # X=X.astype(np.float32)
    # 处理分类特征
    categorical_cols = ['blue', 'dual_sim', 'four_g', 'three_g',
                        'touch_screen', 'wifi', 'resolution_category']

    # 确保分类特征为整数类型
    for col in categorical_cols:
        if col in X.columns:
            X[col] = X[col].astype(np.float32)

    # 划分训练集、验证集和测试集
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp
    )  # 最终比例为 70:15:15

    print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")

    # 特征缩放,Scaler 的作用：把不同量纲、不同取值范围的特征，拉到“同一数值尺度”，让神经网络更容易训练、更稳定、更快收敛。
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 标签编码（已经是0-3）
    y_train_encoded = tf.keras.utils.to_categorical(y_train, 4)
    y_val_encoded = tf.keras.utils.to_categorical(y_val, 4)
    y_test_encoded = tf.keras.utils.to_categorical(y_test, 4)

    joblib.dump(scaler, 'scaler.pkl')

    return (X_train_scaled, X_val_scaled, X_test_scaled,
            y_train_encoded, y_val_encoded, y_test_encoded, scaler)


# 4. 构建ANN模型
def build_ann_model(input_dim, learning_rate=0.001, dropout_rate=0.3):
    """
    构建复杂的ANN模型
    """
    model = keras.Sequential([
        # 输入层
        layers.Input(shape=(input_dim,)),

        # 第一个隐藏层
        layers.Dense(256, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),

        # 第二个隐藏层
        layers.Dense(128, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate * 0.8),

        # 第三个隐藏层
        layers.Dense(64, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate * 0.6),

        # 第四个隐藏层
        layers.Dense(32, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate * 0.4),

        # 输出层（4个类别）
        layers.Dense(4, activation='softmax')
    ])

    # 编译模型
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )

    return model


# 5. 训练模型
def train_model(X_train, y_train, X_val, y_val, input_dim):
    """
    训练ANN模型
    """
    # 创建回调函数
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1
    )

    model_checkpoint = callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

    # 构建模型
    model = build_ann_model(input_dim)

    # 打印模型摘要
    model.summary()

    # 训练模型
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=150,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr, model_checkpoint],
        verbose=1
    )

    return model, history


# 6. 模型评估
def evaluate_model(model, history, X_test, y_test, y_test_raw):
    """
    评估模型性能
    """
    # 加载最佳模型
    try:
        model = keras.models.load_model('best_model.h5')
        print("加载了保存的最佳模型")
    except:
        print("使用最后训练的模型")

    # 在测试集上评估
    test_results = model.evaluate(X_test, y_test, verbose=0)

    print("\n" + "=" * 50)
    print("模型在测试集上的性能:")
    print("=" * 50)
    print(f"测试损失: {test_results[0]:.4f}")
    print(f"测试准确率: {test_results[1]:.4f}")
    print(f"测试精确率: {test_results[2]:.4f}")
    print(f"测试召回率: {test_results[3]:.4f}")
    print(f"测试AUC: {test_results[4]:.4f}")

    # 预测
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # 分类报告
    print("\n分类报告:")
    print(classification_report(y_test_raw, y_pred,
                                target_names=['低端(0)', '中低端(1)', '中高端(2)', '高端(3)']))

    # 混淆矩阵
    cm = confusion_matrix(y_test_raw, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['低端', '中低端', '中高端', '高端'],
                yticklabels=['低端', '中低端', '中高端', '高端'])
    plt.title('混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.show()

    # 训练历史可视化
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 准确率
    axes[0, 0].plot(history.history['accuracy'], label='训练准确率')
    axes[0, 0].plot(history.history['val_accuracy'], label='验证准确率')
    axes[0, 0].set_title('模型准确率')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('准确率')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 损失
    axes[0, 1].plot(history.history['loss'], label='训练损失')
    axes[0, 1].plot(history.history['val_loss'], label='验证损失')
    axes[0, 1].set_title('模型损失')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('损失')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # 精确率和召回率
    axes[1, 0].plot(history.history['precision'], label='训练精确率')
    axes[1, 0].plot(history.history['val_precision'], label='验证精确率')
    axes[1, 0].set_title('精确率')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('精确率')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].plot(history.history['recall'], label='训练召回率')
    axes[1, 1].plot(history.history['val_recall'], label='验证召回率')
    axes[1, 1].set_title('召回率')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('召回率')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

    return test_results


# 7. 特征重要性分析
def analyze_feature_importance(model, X_train, feature_names):
    """
    分析特征重要性
    """
    # 创建简单的特征重要性分析（基于权重）
    weights = model.layers[0].get_weights()[0]
    importance = np.mean(np.abs(weights), axis=1)

    # 创建特征重要性DataFrame
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(12, 6))
    plt.barh(range(len(feature_importance)), feature_importance['importance'])
    plt.yticks(range(len(feature_importance)), feature_importance['feature'])
    plt.xlabel('特征重要性')
    plt.title('ANN模型特征重要性')
    plt.tight_layout()
    plt.show()

    return feature_importance


# 8. 集成学习增强
def build_ensemble_model(input_dim):
    """
    构建集成ANN模型
    """
    models = []

    for i in range(3):
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(4, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        models.append(model)

    return models

def predict_new_phone(model, scaler, feature_names, phone_features):
    """
    预测新手机的价格区间
    """
    # 转换为DataFrame
    phone_df = pd.DataFrame([phone_features])

    # 确保所有特征都存在
    # for feature in feature_names:
    #     if feature not in phone_df.columns:
    #         phone_df[feature] = 0
    #
    # # 重新排序列以匹配训练数据
    # phone_df = phone_df[feature_names]

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


# 9. 主函数
def main():
    """
    主执行函数
    """
    print("=" * 60)
    print("手机价格分类 - ANN模型训练")
    print("=" * 60)

    # 1. 加载和探索数据
    print("\n1. 加载和探索数据...")
    df = load_and_explore_data()

    # 2. 特征工程
    print("\n2. 特征工程...")
    df_engineered = feature_engineering(df)
    print(f"特征工程后形状: {df_engineered.shape}")
    print(f"新特征: {[col for col in df_engineered.columns if col not in df.columns]}")

    # 3. 数据预处理
    print("\n3. 数据预处理...")
    (X_train, X_val, X_test,
     y_train, y_val, y_test, scaler) = preprocess_data(df_engineered)

    # 保存原始测试标签用于评估
    y_test_raw = np.argmax(y_test, axis=1)

    # 4. 训练模型
    print("\n4. 训练ANN模型...")
    input_dim = X_train.shape[1]
    model, history = train_model(X_train, y_train, X_val, y_val, input_dim)

    # 5. 评估模型
    print("\n5. 评估模型性能...")
    test_results = evaluate_model(model, history, X_test, y_test, y_test_raw)

    # 6. 特征重要性分析
    print("\n6. 分析特征重要性...")
    feature_names = df_engineered.drop('price_range', axis=1).columns.tolist()
    feature_importance = analyze_feature_importance(model, X_train, feature_names)

    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"最终测试准确率: {test_results[1]:.4f}")
    print("=" * 60)

    # 7. 保存模型
    model.save('mobile_price_classifier.h5')
    print("模型已保存为 'mobile_price_classifier.h5'")

    return model, history, df_engineered


# 10. 运行主函数
if __name__ == "__main__":
    # model, history, df = main()
    # 加载已保存的模型
    model = keras.models.load_model('mobile_price_classifier.h5')
    # ====== 简单测试样例 ======
    test_phone = {
        'battery_power': 4500,
        'blue': 1,
        'clock_speed': 2.8,
        'dual_sim': 1,
        'fc': 32,
        'four_g': 1,
        'int_memory': 128,
        'm_dep': 0.7,
        'mobile_wt': 190,
        'n_cores': 8,
        'pc': 64,
        'ram': 8192,
        'talk_time': 25,
        'three_g': 1,
        'touch_screen': 1,
        'wifi': 1,

        # ====== 特征工程生成的字段 ======
        'total_pixels': 2400 * 1080,
        'screen_area': 16 * 7,
        'pixel_density': (2400 * 1080) / (16 * 7),
        'battery_to_weight': 4500 / 190,
        'total_cameras': 32 + 64,
        'memory_to_ram': 128 / (8192 / 1024),
        'network_score': 1 + 1 + 1 + 1,
        'processor_power': 2.8 * 8,
        'battery_efficiency': 25 / 4500 * 1000,
        'resolution_category': 3
    }
    scaler = joblib.load('scaler.pkl')

    # 特征顺序（必须和训练一致）
    feature_names = [
        'battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',
        'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'ram',
        'talk_time', 'three_g', 'touch_screen', 'wifi',
        'total_pixels', 'screen_area', 'pixel_density', 'battery_to_weight',
        'total_cameras', 'memory_to_ram', 'network_score', 'processor_power',
        'battery_efficiency', 'resolution_category'
    ]

    # 构造 DataFrame,把无序的输入参数，转换为“顺序、结构、语义”都与训练阶段完全一致的模型输入。
    test_df = pd.DataFrame([test_phone])[feature_names]

    # 标准化,用“训练时学到的标准化规则”转换到模型熟悉的数值空间里
    test_scaled = scaler.transform(test_df)

    # 预测
    probs = model.predict(test_scaled)
    pred = probs.argmax(axis=1)[0]

    price_map = {0: '低端', 1: '中低端', 2: '中高端', 3: '高端'}

    print("预测价格区间:", pred)
    print("价格等级:", price_map[pred])
    print("各类别概率:", probs[0])
