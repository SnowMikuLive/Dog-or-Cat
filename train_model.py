#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
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

def create_data_loaders(data_dir, batch_size=32, img_size=224):
    """创建数据加载器"""
    
    print("[执行] 创建数据加载器...")
    
    # 训练数据变换（包含数据增强）
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 验证数据变换（不包含数据增强）
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    train_dataset = CatDogDataset(data_dir / 'train', transform=train_transform)
    val_dataset = CatDogDataset(data_dir / 'validation', transform=val_transform)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"[OK] 训练样本: {len(train_dataset)}")
    print(f"[OK] 验证样本: {len(val_dataset)}")
    
    return train_loader, val_loader

def create_model(num_classes=2, pretrained=True):
    """创建ResNet模型"""
    
    print("[执行] 创建模型...")
    
    # 使用预训练的ResNet18
    model = models.resnet18(pretrained=pretrained)
    
    # 修改最后一层以适应二分类
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    print("[OK] 模型创建完成")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    return model

def train_model(model, train_loader, val_loader, num_epochs=30, learning_rate=0.001):
    """训练模型"""
    
    print("[执行] 开始训练模型...")
    
    # 检查CUDA可用性
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"[OK] 使用CUDA设备: {torch.cuda.get_device_name()}")
        print(f"CUDA内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device('cpu')
        print("[警告] CUDA不可用，使用CPU训练")
    
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # 记录训练历史
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    best_val_acc = 0.0
    best_model_path = 'models/best_pytorch_model.pth'
    
    # 创建模型目录
    Path('models').mkdir(exist_ok=True)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        train_bar = tqdm(train_loader, desc="训练")
        for inputs, labels in train_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            train_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{running_corrects.double() / ((train_bar.n + 1) * inputs.size(0)):.4f}'
            })
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc.cpu().numpy())
        
        # 验证阶段
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc="验证")
            for inputs, labels in val_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)
                
                val_bar.set_postfix({
                    'val_loss': f'{loss.item():.4f}',
                    'val_acc': f'{val_running_corrects.double() / ((val_bar.n + 1) * inputs.size(0)):.4f}'
                })
        
        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_acc = val_running_corrects.double() / len(val_loader.dataset)
        
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_acc.cpu().numpy())
        
        print(f"训练 - 损失: {epoch_loss:.4f}, 准确率: {epoch_acc:.4f}")
        print(f"验证 - 损失: {val_epoch_loss:.4f}, 准确率: {val_epoch_acc:.4f}")
        
        # 保存最佳模型
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"[OK] 保存最佳模型，准确率: {best_val_acc:.4f}")
        
        scheduler.step()
        
        # 显示当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f"学习率: {current_lr:.6f}")
    
    print(f"\n[OK] 训练完成！最佳验证准确率: {best_val_acc:.4f}")
    
    # 绘制训练历史
    plot_training_history(train_losses, train_accuracies, val_losses, val_accuracies)
    
    return model, {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_acc': best_val_acc
    }

def plot_training_history(train_losses, train_accs, val_losses, val_accs):
    """绘制训练历史"""
    
    print("[执行] 绘制训练历史...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # 绘制准确率
    ax1.plot(epochs, train_accs, 'b-', label='训练准确率')
    ax1.plot(epochs, val_accs, 'r-', label='验证准确率')
    ax1.set_title('模型准确率')
    ax1.set_xlabel('轮次')
    ax1.set_ylabel('准确率')
    ax1.legend()
    ax1.grid(True)
    
    # 绘制损失
    ax2.plot(epochs, train_losses, 'b-', label='训练损失')
    ax2.plot(epochs, val_losses, 'r-', label='验证损失')
    ax2.set_title('模型损失')
    ax2.set_xlabel('轮次')
    ax2.set_ylabel('损失')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    # 创建模型目录
    Path('models').mkdir(exist_ok=True)
    
    plt.savefig('models/pytorch_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("[OK] 训练历史图已保存到 models/pytorch_training_history.png")

def main():
    """主函数"""
    
    print("============================================================")
    print("[执行] PyTorch 猫狗分类模型训练")
    print("============================================================")
    
    # 检查数据目录
    data_dir = Path('data/organized')
    if not data_dir.exists():
        print(f"[错误] 数据目录不存在: {data_dir}")
        print("请先运行数据预处理脚本")
        return
    
    # 检查训练和验证目录
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'validation'
    
    if not train_dir.exists() or not val_dir.exists():
        print("[错误] 训练或验证目录不存在")
        print("请先运行数据预处理脚本")
        return
    
    try:
        # 创建数据加载器
        train_loader, val_loader = create_data_loaders(data_dir, batch_size=32, img_size=224)
        
        if len(train_loader.dataset) == 0 or len(val_loader.dataset) == 0:
            print("[错误] 数据集为空，请检查数据目录")
            return
        
        # 创建模型
        model = create_model(num_classes=2, pretrained=True)
        
        # 训练模型
        trained_model, history = train_model(
            model, 
            train_loader, 
            val_loader, 
            num_epochs=30, 
            learning_rate=0.001
        )
        
        print("[OK] 训练完成！")
        print(f"最佳验证准确率: {history['best_val_acc']:.4f}")
        
    except Exception as e:
        print(f"[错误] 训练过程中出现错误: {e}")
        return

if __name__ == "__main__":
    main() 