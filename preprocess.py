"""
preprocess_kdd_xiiot.py
联合处理 KDD-99 和 X-IIoTID 数据集，为 BERT 微调准备输入文本。
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# 设置随机种子
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# 文件路径（请根据实际情况修改）
KDD_PATH = "data/kdd99.csv"          # KDD-99 数据集文件
XIIOT_PATH = "data/xiiotid.csv"       # X-IIoTID 数据集文件
OUTPUT_DIR = "processed_data_bert"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 采样数量
SAMPLE_SIZE = 100000  # 每个数据集采样10万条
TEST_SIZE_PER_DATASET = 10000  # 每个数据集留出1万条测试

# 特殊分隔符
SEP = " [SEP] "
CLS = "[CLS] "

def process_kdd(df):
    """处理 KDD-99 数据：标签二值化，构造文本"""
    # 标签映射：normal -> 0, 其他 -> 1
    df['label'] = df['label'].apply(lambda x: 0 if x.strip() == 'normal' else 1)
    
    # 获取所有特征列（排除标签列）
    feature_cols = [col for col in df.columns if col != 'label']
    
    # 构造文本：对于每一行，将特征名和值用等号连接，再用[SEP]分隔
    texts = []
    for idx, row in df.iterrows():
        features = [f"{col}={row[col]}" for col in feature_cols]
        text = CLS + SEP.join(features) + SEP + f"label={row['label']}"
        texts.append(text)
    df['text'] = texts
    return df[['text', 'label']]

def process_xiiot(df):
    """处理 X-IIoTID 数据：标签二值化，处理缺失值，构造文本"""
    # 标签映射：class3 中 Normal -> 0, 其他 -> 1
    df['label'] = df['class3'].apply(lambda x: 0 if x.strip() == 'Normal' else 1)
    
    # 删除原始标签列，只保留特征列
    feature_cols = [col for col in df.columns if col not in ['label', 'class1', 'class2', 'class3', 'Date', 'Timestamp', 'Scr_IP', 'Des_IP']]
    # 注意：IP地址和时间戳也可能包含信息，但为简化，我们可以保留它们作为特征。这里选择保留所有列，但处理缺失值。
    # 实际上，我们应保留所有原始列（包括IP等），因为它们可能是重要特征。但为了统一，我们保留所有列（除了标签和多余的分类列）。
    # 更稳妥：保留所有数值和符号列，包括IP、时间戳等，但需处理为字符串。
    # 这里我们保留除class1,class2,class3外的所有列。
    feature_cols = [col for col in df.columns if col not in ['class1', 'class2', 'class3', 'label']]
    
    # 处理缺失值：将 '-' 替换为 'missing'
    df = df.replace('-', 'missing')
    
    # 构造文本
    texts = []
    for idx, row in df.iterrows():
        features = []
        for col in feature_cols:
            val = row[col]
            # 将值转为字符串，如果是数值则直接保留
            features.append(f"{col}={val}")
        text = CLS + SEP.join(features) + SEP + f"label={row['label']}"
        texts.append(text)
    df['text'] = texts
    return df[['text', 'label']]

def main():
    # 1. 加载数据
    print("加载 KDD-99...")
    kdd_df = pd.read_csv(KDD_PATH)
    print(f"KDD-99 原始大小: {len(kdd_df)}")
    
    print("加载 X-IIoTID...")
    xiiot_df = pd.read_csv(XIIOT_PATH)
    print(f"X-IIoTID 原始大小: {len(xiiot_df)}")
    
    # 2. 采样（可选，如果数据量太大）
    if len(kdd_df) > SAMPLE_SIZE:
        kdd_df = kdd_df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)
    if len(xiiot_df) > SAMPLE_SIZE:
        xiiot_df = xiiot_df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)
    
    # 3. 分别处理
    print("处理 KDD-99...")
    kdd_processed = process_kdd(kdd_df)
    print("处理 X-IIoTID...")
    xiiot_processed = process_xiiot(xiiot_df)
    
    # 4. 划分测试集（每个数据集独立留出）
    kdd_train_val, kdd_test = train_test_split(
        kdd_processed, test_size=TEST_SIZE_PER_DATASET, random_state=RANDOM_SEED, stratify=kdd_processed['label']
    )
    xiiot_train_val, xiiot_test = train_test_split(
        xiiot_processed, test_size=TEST_SIZE_PER_DATASET, random_state=RANDOM_SEED, stratify=xiiot_processed['label']
    )
    
    # 5. 合并训练+验证集，并划分
    train_val = pd.concat([kdd_train_val, xiiot_train_val], ignore_index=True)
    train, val = train_test_split(
        train_val, test_size=0.2, random_state=RANDOM_SEED, stratify=train_val['label']
    )
    
    # 6. 合并测试集（两个数据集的测试集合并）
    test = pd.concat([kdd_test, xiiot_test], ignore_index=True)
    
    # 7. 保存
    train.to_csv(os.path.join(OUTPUT_DIR, 'train.csv'), index=False)
    val.to_csv(os.path.join(OUTPUT_DIR, 'val.csv'), index=False)
    test.to_csv(os.path.join(OUTPUT_DIR, 'test.csv'), index=False)
    
    print(f"训练集大小: {len(train)}")
    print(f"验证集大小: {len(val)}")
    print(f"测试集大小: {len(test)}")
    print("预处理完成！")

if __name__ == "__main__":
    main()