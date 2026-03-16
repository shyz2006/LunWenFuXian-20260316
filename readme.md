# BERT + LoRA 入侵检测实验 (KDD-99 & X-IIoTID)

本实验基于论文 **BERTECTOR: Intrusion Detection Based on Joint-Dataset Learning** 的核心思想，使用 BERT 模型结合 LoRA 高效微调，在 **KDD-99** 和 **X-IIoTID** 两个异构数据集上进行二分类（正常/攻击）训练，验证联合数据集训练的可行性和模型泛化能力。

## 📁 实验结构

```
.
├── data                         # 原始为空，需要自行下载 
├── preprocess.py                # 数据预处理：标签统一、构造文本、下采样
├── train_bert_lora.py           # 使用 LoRA 微调 BERT 进行二分类 训练时间在1h左右
├── bert_lora_finetuned          # 执行 train_bert_lora.py 后自动生成
├── README.md                    # 本实验说明
├── requirements.txt             # 项目依赖（已生成）
└── processed_data_bert/         # 预处理后生成的数据（运行后自动创建 我应该也上传了）
    ├── train.csv
    ├── val.csv
    └── test.csv
```

## 📊 数据集

- **KDD-99**：网络连接记录，包含约49万条样本，41个特征，标签为`normal`或攻击类型。
- **X-IIoTID**：工业物联网入侵检测数据集，包含约82万条样本，68个特征，标签位于`class3`列（`Normal`表示正常，其余为攻击）。

需要手动创建`data`文件夹，在里面放入原始数据
原始数据获取地址
**KDD-99**: https://www.kaggle.com/datasets/toobajamal/kdd99-dataset/data
**X-IIoTID**: https://www.kaggle.com/datasets/munaalhawawreh/xiiotid-iiot-intrusion-dataset

下载并解压，重命名为kdd99.csv和xiiotid.csv，放入`data`文件夹

**预处理方式**：
- 将两个数据集的标签统一为二分类：正常 → 0，攻击 → 1。
- 将所有特征（包括符号型）转换为 `特征名=值` 的字符串，并用 `[SEP]` 连接，形成适合 BERT 输入的文本。
- 为解决内存问题，对训练集采样至 **20,000** 条，验证集和测试集各采样 **5,000** 条（随机采样，保持类别分布）。

## 🧪 环境配置

本项目使用 **Python 3.10**，建议通过 conda 创建虚拟环境：

```bash
conda create -n bertector python=3.10
conda activate bertector
```

然后安装项目依赖（已通过 `pip freeze > requirements.txt` 生成）：

```bash
pip install -r requirements.txt
```

若需要手动安装关键库，核心依赖如下：
- PyTorch (≥2.0)
- Transformers (≥4.40.0)
- PEFT (≥0.10.0)
- Datasets
- Scikit-learn
- Pandas, NumPy

## 🚀 运行步骤

### 1. 数据预处理
将原始数据集文件（`kdd99.csv` 和 `xiiotid.csv`）放在项目根目录的 `data/` 文件夹下（或修改脚本中的路径），然后运行：
```bash
python preprocess.py
```
执行后会在 `processed_data_bert/` 下生成训练、验证、测试集文件。

### 2. 训练模型
```bash
python train_bert_lora.py
```
训练参数已优化以适配 8GB 显存：
- 批次大小：4（梯度累积 4，等效 16）
- 梯度检查点：开启
- 混合精度（fp16）：开启
- LoRA 秩：8，作用于 `query`、`value` 层
- 训练轮数：3（可根据需要调整）

训练过程中会输出每个 epoch 的验证集指标，并自动保存最佳模型（按 F1 分数）。

### 3. 评估模型
训练结束后脚本会自动在测试集上评估，并打印准确率、精确率、召回率、F1 分数。

## 📈 实验结果（3 epoch，采样后）

| 指标 | 值 |
|------|-----|
| 准确率 (Accuracy) | 87.00% |
| 精确率 (Precision) | 87.12% |
| 召回率 (Recall) | 87.00% |
| F1 分数 (Weighted) | 0.8705 |

验证集 F1 在训练过程中达到约 **0.875**，表明模型能有效区分正常与攻击流量。

## 📌 注意事项

- **内存与显存**：若遇到 `out of memory`，可进一步减小 `BATCH_SIZE` 或增大 `gradient_accumulation_steps`。
- **数据路径**：请确保预处理脚本中的文件路径正确，或根据实际情况修改 `TRAIN_PATH` 等变量。
- **原始数据**：由于文件较大，未包含在仓库中，请自行下载并放置于指定位置。

## 原始论文
详见
https://arxiv.org/abs/2508.10327