"""
train_bert_lora.py
使用 LoRA 微调 BERT 进行二分类入侵检测（极致显存优化 + 内存友好版）
"""

import os
# 设置 Hugging Face 镜像站端点
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# ---------- 显存优化设置 ----------
# 启用 PyTorch 可扩展段以减少显存碎片（如果平台支持）
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# 关闭 Hugging Face Hub 的软链接警告
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# -------------------- 配置 --------------------
MODEL_NAME = "bert-base-uncased"          # BERT 基础模型
OUTPUT_DIR = "./bert_lora_finetuned"
BATCH_SIZE = 4                             # 批次大小
GRADIENT_ACCUMULATION_STEPS = 4            # 梯度累积步数，等效于 batch size 16
LEARNING_RATE = 2e-5
EPOCHS = 3 #没那么多时间 从10轮调整到3轮
MAX_LENGTH = 512                           # BERT 最大输入长度
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- 内存友好配置 ----------
# 由于数据集较大，我们只使用部分数据进行训练和验证
TRAIN_SAMPLE_SIZE = 20000    # 训练集采样数量（原144000）
VAL_SAMPLE_SIZE = 5000       # 验证集采样数量（原36000）
TEST_SAMPLE_SIZE = 5000       # 测试集采样数量（原20000）

# 数据路径
TRAIN_PATH = "processed_data_bert/train.csv"
VAL_PATH = "processed_data_bert/val.csv"
TEST_PATH = "processed_data_bert/test.csv"

# -------------------- 1. 加载数据并进行下采样 --------------------
print("加载数据...")
# 读取全部数据（如果内存不足，可以考虑在读取时指定部分列，但这里我们完整读取后采样）
train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)

print(f"原始训练集大小: {len(train_df)}，验证集大小: {len(val_df)}，测试集大小: {len(test_df)}")

# 随机采样
if len(train_df) > TRAIN_SAMPLE_SIZE:
    train_df = train_df.sample(n=TRAIN_SAMPLE_SIZE, random_state=42)
if len(val_df) > VAL_SAMPLE_SIZE:
    val_df = val_df.sample(n=VAL_SAMPLE_SIZE, random_state=42)
if len(test_df) > TEST_SAMPLE_SIZE:
    test_df = test_df.sample(n=TEST_SAMPLE_SIZE, random_state=42)

print(f"采样后训练集大小: {len(train_df)}，验证集大小: {len(val_df)}，测试集大小: {len(test_df)}")

# 提取文本和标签
train_texts = train_df['text'].tolist()
train_labels = train_df['label'].tolist()
val_texts = val_df['text'].tolist()
val_labels = val_df['label'].tolist()
test_texts = test_df['text'].tolist()
test_labels = test_df['label'].tolist()

# 转换为 HuggingFace Dataset
train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})
test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})

# -------------------- 2. 加载 tokenizer 和 tokenize --------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

print("Tokenizing datasets...")
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# 设置格式（移除文本列，只保留模型需要的列）
train_dataset = train_dataset.remove_columns(["text"])
val_dataset = val_dataset.remove_columns(["text"])
test_dataset = test_dataset.remove_columns(["text"])
train_dataset.set_format("torch")
val_dataset.set_format("torch")
test_dataset.set_format("torch")

# -------------------- 3. 加载基础模型 --------------------
print("加载基础模型...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,                           # 二分类
    ignore_mismatched_sizes=True            # 消除分类头参数缺失警告
)
model.to(DEVICE)

# 启用梯度检查点（gradient checkpointing）以进一步减少显存占用
model.gradient_checkpointing_enable()

# -------------------- 4. LoRA 配置 --------------------
print("配置 LoRA...")
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,             # 序列分类任务
    r=8,                                     # 低秩矩阵的秩（可调整）
    lora_alpha=32,                           # 缩放参数
    target_modules=["query", "value"],       # 对 BERT 的 query 和 value 层施加 LoRA
    lora_dropout=0.1,
    bias="none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# -------------------- 5. 定义评估指标 --------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
    acc = accuracy_score(labels, predictions)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

# -------------------- 6. 训练参数 --------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",            # 每个 epoch 评估一次
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,  # 梯度累积
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    fp16=True,                              # 开启混合精度训练，减少显存占用
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_dir='./logs',
    logging_steps=500,
    save_total_limit=2,
    remove_unused_columns=False,
    dataloader_num_workers=0,                # 禁用多进程数据加载，避免内存不足
)

# 早停回调（可选）
early_stop = EarlyStoppingCallback(early_stopping_patience=3)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[early_stop],
)

# -------------------- 7. 开始训练 --------------------
print("开始训练...")
trainer.train()

# 保存最终模型
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"模型已保存到 {OUTPUT_DIR}")

# -------------------- 8. 在测试集上评估 --------------------
print("在测试集上评估...")
test_results = trainer.predict(test_dataset)
print("测试集结果:")
print(f"准确率: {test_results.metrics['test_accuracy']:.4f}")
print(f"精确率: {test_results.metrics['test_precision']:.4f}")
print(f"召回率: {test_results.metrics['test_recall']:.4f}")
print(f"F1 分数: {test_results.metrics['test_f1']:.4f}")