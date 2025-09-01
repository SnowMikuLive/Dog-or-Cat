#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用训练好的PyTorch模型进行预测
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import argparse

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class CatDogPredictor:
    """猫狗分类预测器"""
    
    def __init__(self, model_path='models/best_pytorch_model.pth'):
        """初始化预测器"""
        self.model_path = model_path
        self.model = None
        self.device = None
        self.img_size = 224
        self.class_names = ['猫', '狗']
        self.transform = self.create_transform()
        self.setup_device()
        self.load_model()
    
    def setup_device(self):
        """设置计算设备"""
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"[OK] 使用CUDA设备: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device('cpu')
            print("[信息] 使用CPU设备")
    
    def create_transform(self):
        """创建图片变换"""
        return transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_model(self):
        """加载模型"""
        print(f"[执行] 加载模型: {self.model_path}")
        
        if not Path(self.model_path).exists():
            print(f"[错误] 模型文件不存在: {self.model_path}")
            return False
        
        try:
            # 创建模型架构
            self.model = models.resnet18(pretrained=False)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, 2)  # 二分类
            
            # 加载模型权重
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print("[OK] 模型加载成功")
            return True
        except Exception as e:
            print(f"[错误] 模型加载失败: {e}")
            return False
    
    def preprocess_image(self, img_path):
        """预处理图片"""
        try:
            # 加载图片
            img = Image.open(img_path).convert('RGB')
            original_img = img.copy()
            
            # 应用变换
            img_tensor = self.transform(img).unsqueeze(0)  # 添加批次维度
            
            return img_tensor, original_img
        
        except Exception as e:
            print(f"[错误] 图片预处理失败: {e}")
            return None, None
    
    def predict_single_image(self, img_path, show_image=True):
        """预测单张图片"""
        if self.model is None:
            print("[错误] 模型未加载")
            return None
        
        print(f"[执行] 预测图片: {img_path}")
        
        # 预处理图片
        img_tensor, original_img = self.preprocess_image(img_path)
        if img_tensor is None:
            return None
        
        # 进行预测
        with torch.no_grad():
            img_tensor = img_tensor.to(self.device)
            outputs = self.model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            predicted_class = predicted.item()
            confidence = probabilities[0][predicted_class].item()
            dog_probability = probabilities[0][1].item()  # 狗的概率
        
        result = {
            'class': predicted_class,
            'class_name': self.class_names[predicted_class],
            'confidence': confidence,
            'dog_probability': dog_probability,
            'cat_probability': probabilities[0][0].item()
        }
        
        print(f"[OK] 预测结果: {result['class_name']} (置信度: {result['confidence']:.4f})")
        
        # 显示图片和预测结果
        if show_image and original_img:
            self.display_prediction(original_img, result, img_path)
        
        return result
    
    def display_prediction(self, img, result, img_path):
        """显示预测结果"""
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.axis('off')
        
        # 设置标题颜色
        color = 'green' if result['confidence'] > 0.8 else 'orange' if result['confidence'] > 0.6 else 'red'
        
        plt.title(
            f"预测: {result['class_name']} (置信度: {result['confidence']:.4f})",
            fontsize=16,
            color=color,
            fontweight='bold'
        )
        
        # 添加概率信息
        prob_text = f"猫: {result['cat_probability']:.3f} | 狗: {result['dog_probability']:.3f}"
        plt.figtext(0.5, 0.08, prob_text, ha='center', fontsize=12)
        
        # 添加文件名
        plt.figtext(0.5, 0.02, f"文件: {Path(img_path).name}", ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.show()
    
    def predict_batch(self, img_dir, output_csv=None):
        """批量预测图片"""
        img_dir = Path(img_dir)
        
        if not img_dir.exists():
            print(f"[错误] 图片目录不存在: {img_dir}")
            return []
        
        # 支持的图片格式
        img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        img_files = []
        
        for ext in img_extensions:
            img_files.extend(img_dir.glob(ext))
            img_files.extend(img_dir.glob(ext.upper()))
        
        if not img_files:
            print(f"[错误] 在目录 {img_dir} 中没有找到图片文件")
            return []
        
        print(f"[信息] 找到 {len(img_files)} 张图片")
        
        results = []
        
        for i, img_file in enumerate(img_files, 1):
            print(f"处理 {i}/{len(img_files)}: {img_file.name}")
            result = self.predict_single_image(img_file, show_image=False)
            if result:
                result['filename'] = img_file.name
                result['filepath'] = str(img_file)
                results.append(result)
        
        # 保存结果到CSV
        if output_csv and results:
            self.save_results_to_csv(results, output_csv)
        
        # 显示统计信息
        self.show_batch_statistics(results)
        
        return results
    
    def save_results_to_csv(self, results, output_csv):
        """保存结果到CSV文件"""
        df_data = []
        for result in results:
            df_data.append({
                '文件名': result['filename'],
                '文件路径': result['filepath'],
                '预测类别': result['class_name'],
                '置信度': result['confidence'],
                '猫概率': result['cat_probability'],
                '狗概率': result['dog_probability']
            })
        
        df = pd.DataFrame(df_data)
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"[OK] 预测结果已保存到: {output_csv}")
    
    def show_batch_statistics(self, results):
        """显示批量预测统计信息"""
        if not results:
            return
        
        total = len(results)
        cats = sum(1 for r in results if r['class'] == 0)
        dogs = sum(1 for r in results if r['class'] == 1)
        
        high_confidence = sum(1 for r in results if r['confidence'] > 0.8)
        medium_confidence = sum(1 for r in results if 0.6 < r['confidence'] <= 0.8)
        low_confidence = sum(1 for r in results if r['confidence'] <= 0.6)
        
        avg_confidence = np.mean([r['confidence'] for r in results])
        
        print("\n[结果] 批量预测统计:")
        print(f"总图片数: {total}")
        print(f"预测为猫: {cats} ({cats/total*100:.1f}%)")
        print(f"预测为狗: {dogs} ({dogs/total*100:.1f}%)")
        print(f"平均置信度: {avg_confidence:.4f}")
        print(f"\n置信度分布:")
        print(f"高置信度 (>80%): {high_confidence} ({high_confidence/total*100:.1f}%)")
        print(f"中等置信度 (60-80%): {medium_confidence} ({medium_confidence/total*100:.1f}%)")
        print(f"低置信度 (<=60%): {low_confidence} ({low_confidence/total*100:.1f}%)")
    
    def predict_test_dataset(self, test_dir='data/test1', output_csv='predictions.csv'):
        """预测测试数据集"""
        print(f"[执行] 预测测试数据集: {test_dir}")
        
        test_path = Path(test_dir)
        if not test_path.exists():
            print(f"[错误] 测试目录不存在: {test_dir}")
            return []
        
        # 获取所有图片文件并按文件名排序
        img_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            img_files.extend(test_path.glob(ext))
        
        if not img_files:
            print("[错误] 测试目录中没有找到图片文件")
            return []
        
        # 按数字顺序排序（如果文件名是数字）
        try:
            img_files.sort(key=lambda x: int(x.stem))
        except ValueError:
            img_files.sort()  # 如果不是数字，按字母顺序排序
        
        print(f"[信息] 找到 {len(img_files)} 张测试图片")
        
        results = []
        predictions_data = []
        
        for i, img_file in enumerate(img_files, 1):
            if i % 100 == 0:  # 每100张显示进度
                print(f"已处理: {i}/{len(img_files)}")
            
            result = self.predict_single_image(img_file, show_image=False)
            if result:
                results.append(result)
                
                # 为Kaggle提交格式准备数据
                img_id = img_file.stem  # 文件名（不含扩展名）
                label = result['dog_probability']  # Kaggle通常需要狗的概率
                
                predictions_data.append({
                    'id': img_id,
                    'label': label
                })
        
        # 保存Kaggle提交格式的结果
        if predictions_data:
            df = pd.DataFrame(predictions_data)
            df.to_csv(output_csv, index=False)
            print(f"[OK] Kaggle提交文件已保存到: {output_csv}")
            
            # 显示前几行预览
            print(f"\n预测结果预览:")
            print(df.head(10))
        
        return results

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='猫狗分类预测')
    parser.add_argument('--model', default='models/best_pytorch_model.pth', help='模型文件路径')
    parser.add_argument('--image', help='单张图片路径')
    parser.add_argument('--batch', help='批量预测图片目录')
    parser.add_argument('--test', action='store_true', help='预测测试数据集')
    parser.add_argument('--output', default='predictions.csv', help='输出CSV文件')
    
    args = parser.parse_args()
    
    print("============================================================")
    print("[执行] PyTorch 猫狗分类预测")
    print("============================================================")
    
    # 创建预测器
    predictor = CatDogPredictor(args.model)
    
    if not predictor.model:
        print("[错误] 模型加载失败，退出程序")
        return
    
    if args.image:
        # 单张图片预测
        result = predictor.predict_single_image(args.image)
        if result:
            print(f"\n[结果] {args.image}")
            print(f"预测类别: {result['class_name']}")
            print(f"置信度: {result['confidence']:.4f}")
            print(f"猫概率: {result['cat_probability']:.4f}")
            print(f"狗概率: {result['dog_probability']:.4f}")
    
    elif args.batch:
        # 批量预测
        results = predictor.predict_batch(args.batch, args.output)
        print(f"\n[OK] 批量预测完成，共处理 {len(results)} 张图片")
    
    elif args.test:
        # 预测测试数据集
        results = predictor.predict_test_dataset(output_csv=args.output)
        print(f"\n[OK] 测试数据集预测完成，共处理 {len(results)} 张图片")
    
    else:
        # 交互式模式
        print("\n[信息] 进入交互式预测模式")
        print("输入图片路径进行预测，输入 'quit' 退出")
        
        while True:
            img_path = input("\n请输入图片路径: ").strip()
            
            if img_path.lower() in ['quit', 'exit', 'q']:
                break
            
            if not img_path:
                continue
            
            if not Path(img_path).exists():
                print(f"[错误] 文件不存在: {img_path}")
                continue
            
            result = predictor.predict_single_image(img_path)
            if result:
                print(f"预测结果: {result['class_name']} (置信度: {result['confidence']:.4f})")

if __name__ == "__main__":
    main() 