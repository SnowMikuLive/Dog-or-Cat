#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理脚本
"""

import os
import shutil
import random
from pathlib import Path
from PIL import Image
import numpy as np

def organize_data():
    """整理数据集结构"""
    
    data_dir = Path('data')
    train_dir = data_dir / 'train'
    test_dir = data_dir / 'test'
    
    # 创建整理后的目录结构
    organized_dir = data_dir / 'organized'
    
    # 创建训练和验证目录
    for split in ['train', 'validation']:
        for category in ['cats', 'dogs']:
            (organized_dir / split / category).mkdir(parents=True, exist_ok=True)
    
    # 创建测试目录
    (organized_dir / 'test').mkdir(parents=True, exist_ok=True)
    
    print("[整理] 开始整理训练数据...")
    
    # 整理训练数据
    if train_dir.exists():
        train_images = list(train_dir.glob('*.jpg'))
        print(f"找到 {len(train_images)} 张训练图片")
        
        # 分离猫和狗的图片
        cat_images = [img for img in train_images if 'cat' in img.name]
        dog_images = [img for img in train_images if 'dog' in img.name]
        
        print(f"猫的图片: {len(cat_images)} 张")
        print(f"狗的图片: {len(dog_images)} 张")
        
        # 随机划分训练集和验证集 (80:20)
        def split_data(images, category):
            random.shuffle(images)
            split_idx = int(0.8 * len(images))
            
            train_imgs = images[:split_idx]
            val_imgs = images[split_idx:]
            
            # 复制到相应目录
            for img in train_imgs:
                dst = organized_dir / 'train' / category / img.name
                shutil.copy2(img, dst)
            
            for img in val_imgs:
                dst = organized_dir / 'validation' / category / img.name
                shutil.copy2(img, dst)
                
            return len(train_imgs), len(val_imgs)
        
        cat_train, cat_val = split_data(cat_images, 'cats')
        dog_train, dog_val = split_data(dog_images, 'dogs')
        
        print(f"[OK] 训练集: 猫 {cat_train} 张, 狗 {dog_train} 张")
        print(f"[OK] 验证集: 猫 {cat_val} 张, 狗 {dog_val} 张")
    
    # 整理测试数据
    if test_dir.exists():
        test_images = list(test_dir.glob('*.jpg'))
        print(f"\n[复制] 复制测试数据: {len(test_images)} 张图片")
        
        for img in test_images:
            dst = organized_dir / 'test' / img.name
            shutil.copy2(img, dst)
        
        print("[OK] 测试数据复制完成")
    
    return organized_dir

def check_data_quality(data_dir):
    """检查数据质量"""
    
    print("\n[检查] 检查数据质量...")
    
    corrupted_images = []
    
    for img_path in data_dir.rglob('*.jpg'):
        try:
            with Image.open(img_path) as img:
                img.verify()  # 验证图片
        except Exception as e:
            print(f"损坏的图片: {img_path} - {e}")
            corrupted_images.append(img_path)
    
    if corrupted_images:
        print(f"[警告] 发现 {len(corrupted_images)} 张损坏的图片")
        # 删除损坏的图片
        for img_path in corrupted_images:
            img_path.unlink()
        print("[OK] 已删除损坏的图片")
    else:
        print("[OK] 所有图片都是完整的")

def get_data_statistics(data_dir):
    """获取数据统计信息"""
    
    print("\n[统计] 数据统计信息:")
    
    for split in ['train', 'validation']:
        split_dir = data_dir / split
        if split_dir.exists():
            print(f"\n{split.upper()}:")
            for category in ['cats', 'dogs']:
                category_dir = split_dir / category
                if category_dir.exists():
                    count = len(list(category_dir.glob('*.jpg')))
                    print(f"  {category}: {count} 张图片")
    
    # 测试集统计
    test_dir = data_dir / 'test'
    if test_dir.exists():
        test_count = len(list(test_dir.glob('*.jpg')))
        print(f"\nTEST: {test_count} 张图片")

def main():
    """主函数"""
    
    print("[开始] 开始数据预处理...")
    
    # 设置随机种子
    random.seed(42)
    
    # 整理数据
    organized_dir = organize_data()
    
    # 检查数据质量
    check_data_quality(organized_dir)
    
    # 获取统计信息
    get_data_statistics(organized_dir)
    
    print("\n[完成] 数据预处理完成!")
    print(f"整理后的数据保存在: {organized_dir}")

if __name__ == "__main__":
    main() 