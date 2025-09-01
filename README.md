# 🐱🐶 猫狗分类项目 - 快速开始指南

## 🚀 项目简介
这是一个完整的深度学习项目，用于进行猫狗图片二分类。项目使用了您提供的Kaggle API凭证来下载数据集，并使用迁移学习（VGG16）来训练模型。

## 📋 项目特点
- ✅ **完全自动化**: 从数据下载到模型训练的完整流程
- ✅ **迁移学习**: 使用预训练的VGG16模型，训练效果更好
- ✅ **数据增强**: 自动进行图像增强，提高模型泛化能力
- ✅ **模型评估**: 完整的评估报告和可视化图表
- ✅ **易于使用**: 图形化界面和命令行工具
- ✅ **批量预测**: 支持单张和批量图片预测

## 🛠️ 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 运行项目
```bash
python run_project.py
```
### 3.分步运行(可选)
1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 配置Kaggle API：
```bash
python setup_kaggle.py
```

3. 下载数据：
```bash
python download_data.py
```

4. 训练模型：
```bash
python train_model.py
```

5. 评估模型：
```bash
python evaluate_model.py
```

6. 进行预测：
```bash
python predict.py
```
### 3. 按照菜单提示操作
项目提供了交互式菜单，您可以选择：
- **选项1**: 安装依赖
- **选项2**: 配置Kaggle API（使用您提供的凭证）
- **选项3**: 下载数据集
- **选项4**: 数据预处理
- **选项5**: 训练模型
- **选项6**: 评估模型
- **选项7**: 使用模型预测
- **选项8**: 运行完整流程
- **选项9**: 检查项目状态

## 📊 预期结果
- **数据集**: 约25,000张猫狗图片
- **模型准确率**: 预期达到85-95%
- **训练时间**: 30分钟-2小时（取决于硬件）
- **输出文件**: 
  - 训练好的模型文件（.h5格式）
  - 训练历史图表
  - 混淆矩阵
  - 预测结果CSV文件

## 🎯 使用模型进行预测

### 单张图片预测
```bash
python predict.py --image "图片路径.jpg"
```

### 批量预测
```bash
python predict.py --batch "图片目录路径" --output "结果.csv"
```

### 交互式预测
```bash
python predict.py
```

## 📁 项目结构
```
Dog or Cat/
├── requirements.txt          # Python依赖包
├── setup_kaggle.py          # Kaggle API配置
├── download_data.py         # 数据下载脚本
├── data_preprocessing.py    # 数据预处理
├── train_model.py          # 模型训练
├── evaluate_model.py       # 模型评估
├── predict.py              # 模型预测
├── run_project.py          # 主启动脚本
├── README.md               # 项目说明
├── 快速开始.md             # 本文件
├── data/                   # 数据目录（运行后生成）
│   ├── train/             # 原始训练数据
│   ├── test/              # 原始测试数据
│   └── organized/         # 整理后的数据
│       ├── train/         # 训练集（猫/狗分类）
│       ├── validation/    # 验证集（猫/狗分类）
│       └── test/          # 测试集
└── models/                # 模型目录（训练后生成）
    ├── best_model.h5      # 最佳模型
    ├── fine_tuned_model.h5 # 微调模型
    ├── final_model.h5     # 最终模型
    └── *.png              # 训练图表
```

## ⚡ 一键运行完整流程
如果您想要一次性完成所有步骤，请：

1. 运行 `python run_project.py`
2. 选择选项 **8**（运行完整流程）
3. 等待完成

## 🔧 故障排除

### 常见问题
1. **Kaggle API错误**: 确保网络连接正常，API凭证正确
2. **内存不足**: 可以在`train_model.py`中调小batch_size
3. **训练时间过长**: 可以减少训练轮次（epochs）
4. **模型文件过大**: 正常现象，VGG16模型约500MB

### 系统要求
- **Python**: 3.7+ - 3.12.X，不支持3.13
- **内存**: 建议8GB+
- **存储**: 至少5GB可用空间
- **GPU**: 可选，但会大大加速训练

## 📞 技术支持
如果遇到问题，请检查：
1. Python版本是否兼容
2. 所有依赖是否正确安装
3. 网络连接是否正常
4. 磁盘空间是否充足

祝您使用愉快！🎉 