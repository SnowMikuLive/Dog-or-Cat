#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用PyTorch评估猫狗分类模型
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class CatDogDataset(Dataset):
    """猫狗数据集类"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        
        # 加载猫的图片 (标签0)
        cat_dir = self.data_dir / 'cats'
        if cat_dir.exists():
            for img_path in cat_dir.glob('*.jpg'):
                self.images.append(str(img_path))
                self.labels.append(0)
        
        # 加载狗的图片 (标签1)
        dog_dir = self.data_dir / 'dogs'
        if dog_dir.exists():
            for img_path in dog_dir.glob('*.jpg'):
                self.images.append(str(img_path))
                self.labels.append(1)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # 加载图片
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"[错误] 无法加载图片 {img_path}: {e}")
            # 返回一个默认的黑色图片
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def load_model(model_path, num_classes=2):
    """加载训练好的PyTorch模型"""
    
    print(f"[执行] 加载模型: {model_path}")
    
    if not Path(model_path).exists():
        print(f"[错误] 模型文件不存在: {model_path}")
        return None
    
    try:
        # 创建模型架构
        model = models.resnet18(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        
        # 加载模型权重
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        
        print("[OK] 模型加载成功")
        return model
    except Exception as e:
        print(f"[错误] 模型加载失败: {e}")
        return None

def create_test_loader(data_dir, img_size=224, batch_size=32):
    """创建测试数据加载器"""
    
    print("[执行] 创建测试数据加载器...")
    
    # 测试数据变换（不包含数据增强）
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = CatDogDataset(data_dir / 'validation', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"[OK] 测试样本: {len(test_dataset)}")
    
    return test_loader

def evaluate_model(model, test_loader, device):
    """评估模型性能"""
    
    print("[执行] 开始评估模型...")
    
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_labels = []
    
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc="评估")
        for inputs, labels in test_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # 狗的概率
            all_labels.extend(labels.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_labels = np.array(all_labels)
    
    # 计算各种指标
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    
    print("\n[结果] 模型评估结果:")
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    
    # 详细分类报告
    class_names = ['cats', 'dogs']
    report = classification_report(
        all_labels, 
        all_predictions, 
        target_names=class_names,
        digits=4
    )
    
    print("\n[结果] 详细分类报告:")
    print(report)
    
    return {
        'predictions': all_predictions,
        'probabilities': all_probabilities,
        'true_labels': all_labels,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'report': report
    }

def plot_confusion_matrix(true_labels, predicted_classes, save_path='models/confusion_matrix.png'):
    """绘制混淆矩阵"""
    
    print("[执行] 绘制混淆矩阵...")
    
    # 计算混淆矩阵
    cm = confusion_matrix(true_labels, predicted_classes)
    
    # 绘制热力图
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['猫', '狗'],
        yticklabels=['猫', '狗']
    )
    
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    
    # 添加准确率信息
    accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
    plt.figtext(0.02, 0.02, f'总体准确率: {accuracy:.4f}', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"[OK] 混淆矩阵已保存到: {save_path}")

def plot_prediction_distribution(probabilities, true_labels, save_path='models/prediction_distribution.png'):
    """绘制预测概率分布"""
    
    print("[执行] 绘制预测概率分布...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 分别绘制猫和狗的预测概率分布
    cat_probabilities = probabilities[true_labels == 0]
    dog_probabilities = probabilities[true_labels == 1]
    
    # 猫的预测分布
    ax1.hist(cat_probabilities, bins=50, alpha=0.7, color='blue', label='猫')
    ax1.axvline(x=0.5, color='red', linestyle='--', label='决策边界')
    ax1.set_title('猫的预测概率分布')
    ax1.set_xlabel('预测概率 (狗的概率)')
    ax1.set_ylabel('频次')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 狗的预测分布
    ax2.hist(dog_probabilities, bins=50, alpha=0.7, color='orange', label='狗')
    ax2.axvline(x=0.5, color='red', linestyle='--', label='决策边界')
    ax2.set_title('狗的预测概率分布')
    ax2.set_xlabel('预测概率 (狗的概率)')
    ax2.set_ylabel('频次')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"[OK] 预测分布图已保存到: {save_path}")

def analyze_errors(results, num_errors=10):
    """分析错误预测的样本"""
    
    print(f"[执行] 分析前{num_errors}个错误预测样本...")
    
    probabilities = results['probabilities']
    predicted_classes = results['predictions']
    true_labels = results['true_labels']
    
    # 找到错误预测的索引
    wrong_indices = np.where(predicted_classes != true_labels)[0]
    
    if len(wrong_indices) == 0:
        print("[OK] 没有错误预测！模型表现完美！")
        return
    
    print(f"[信息] 总共有 {len(wrong_indices)} 个错误预测")
    
    # 显示一些错误样本的信息
    class_names = ['猫', '狗']
    
    print(f"\n前{min(num_errors, len(wrong_indices))}个错误样本:")
    for i, idx in enumerate(wrong_indices[:num_errors]):
        true_class = class_names[true_labels[idx]]
        pred_class = class_names[predicted_classes[idx]]
        confidence = probabilities[idx] if predicted_classes[idx] == 1 else 1 - probabilities[idx]
        
        print(f"样本 {idx}: 真实={true_class}, 预测={pred_class}, 置信度={confidence:.4f}")

def save_results(results, save_path='models/evaluation_results.txt'):
    """保存评估结果到文件"""
    
    print(f"[执行] 保存评估结果到: {save_path}")
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("猫狗分类模型评估结果\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"准确率 (Accuracy): {results['accuracy']:.4f}\n")
        f.write(f"精确率 (Precision): {results['precision']:.4f}\n")
        f.write(f"召回率 (Recall): {results['recall']:.4f}\n")
        f.write(f"F1分数: {results['f1']:.4f}\n\n")
        
        f.write("详细分类报告:\n")
        f.write("-" * 30 + "\n")
        f.write(results['report'])
    
    print("[OK] 评估结果已保存")

def main():
    """主函数"""
    
    print("============================================================")
    print("[执行] PyTorch 猫狗分类模型评估")
    print("============================================================")
    
    # 检查模型文件
    model_path = 'models/best_pytorch_model.pth'
    if not Path(model_path).exists():
        print(f"[错误] 模型文件不存在: {model_path}")
        print("请先训练模型")
        return
    
    # 检查数据目录
    data_dir = Path('data/organized')
    if not data_dir.exists():
        print(f"[错误] 数据目录不存在: {data_dir}")
        print("请先运行数据预处理脚本")
        return
    
    # 检查验证目录
    val_dir = data_dir / 'validation'
    if not val_dir.exists():
        print("[错误] 验证目录不存在")
        print("请先运行数据预处理脚本")
        return
    
    try:
        # 设置设备
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"[OK] 使用CUDA设备: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            print("[信息] 使用CPU设备")
        
        # 加载模型
        model = load_model(model_path)
        if model is None:
            return
        
        model = model.to(device)
        
        # 创建测试数据加载器
        test_loader = create_test_loader(data_dir, img_size=224, batch_size=32)
        
        if len(test_loader.dataset) == 0:
            print("[错误] 测试数据集为空，请检查数据目录")
            return
        
        # 评估模型
        results = evaluate_model(model, test_loader, device)
        
        # 创建模型目录
        Path('models').mkdir(exist_ok=True)
        
        # 绘制混淆矩阵
        plot_confusion_matrix(
            results['true_labels'], 
            results['predictions'],
            'models/pytorch_confusion_matrix.png'
        )
        
        # 绘制预测概率分布
        plot_prediction_distribution(
            results['probabilities'],
            results['true_labels'],
            'models/pytorch_prediction_distribution.png'
        )
        
        # 分析错误预测
        analyze_errors(results, num_errors=10)
        
        # 保存结果
        save_results(results, 'models/pytorch_evaluation_results.txt')
        
        print("\n[OK] 模型评估完成！")
        print(f"最终准确率: {results['accuracy']:.4f}")
        print("评估结果已保存到 models/ 目录下")
        
    except Exception as e:
        print(f"[错误] 评估过程中出现错误: {e}")
        return

if __name__ == "__main__":
    main() 