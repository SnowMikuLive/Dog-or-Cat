#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
猫狗分类器UI界面
使用训练好的模型进行图片识别，显示结果和置信度
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np
import cv2
import os
import threading

class CatDogClassifierUI:
    def __init__(self, root):
        self.root = root
        self.root.title("猫狗分类器 - AI图像识别")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        self.root.minsize(1200, 800)  # 设置最小窗口大小
        
        # 初始化变量
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.current_image = None
        self.original_image = None
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.setup_ui()
        self.load_model()
        
    def setup_ui(self):
        """设置UI界面"""
        # 主标题
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        title_frame.pack(fill='x', pady=(0, 10))
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame, 
            text="AI猫狗分类器", 
            font=('Arial', 24, 'bold'),
            fg='white', 
            bg='#2c3e50'
        )
        title_label.pack(expand=True)
        
        # 主内容框架
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # 左侧控制面板
        control_frame = tk.Frame(main_frame, bg='#ecf0f1', width=350)
        control_frame.pack(side='left', fill='y', padx=(0, 15))
        control_frame.pack_propagate(False)
        
        # 控制面板标题
        control_title = tk.Label(
            control_frame, 
            text="控制面板", 
            font=('Arial', 16, 'bold'),
            bg='#ecf0f1',
            fg='#2c3e50'
        )
        control_title.pack(pady=20)
        
        # 模型状态
        self.model_status_label = tk.Label(
            control_frame,
            text="[加载中] 正在加载模型...",
            font=('Arial', 10),
            bg='#ecf0f1',
            fg='#e74c3c'
        )
        self.model_status_label.pack(pady=10)
        
        # 选择图片按钮
        self.select_btn = tk.Button(
            control_frame,
            text="选择图片",
            font=('Arial', 12, 'bold'),
            bg='#3498db',
            fg='white',
            command=self.select_image,
            relief='flat',
            padx=20,
            pady=10
        )
        self.select_btn.pack(pady=20)
        
        # 识别按钮
        self.classify_btn = tk.Button(
            control_frame,
            text="开始识别",
            font=('Arial', 12, 'bold'),
            bg='#27ae60',
            fg='white',
            command=self.classify_image,
            relief='flat',
            padx=20,
            pady=10,
            state='disabled'
        )
        self.classify_btn.pack(pady=10)
        
        # 批量识别按钮
        self.batch_btn = tk.Button(
            control_frame,
            text="批量识别",
            font=('Arial', 12, 'bold'),
            bg='#8e44ad',
            fg='white',
            command=self.batch_classify,
            relief='flat',
            padx=20,
            pady=10
        )
        self.batch_btn.pack(pady=5)
        
        # 进度条（用于批量处理）
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            control_frame,
            variable=self.progress_var,
            maximum=100,
            style='Processing.Horizontal.TProgressbar'
        )
        self.progress_bar.pack(fill='x', padx=20, pady=5)
        self.progress_bar.pack_forget()  # 初始隐藏
        
        # 批量处理状态标签
        self.batch_status_label = tk.Label(
            control_frame,
            text="",
            font=('Arial', 9),
            bg='#ecf0f1',
            fg='#7f8c8d'
        )
        self.batch_status_label.pack(pady=5)
        self.batch_status_label.pack_forget()  # 初始隐藏
        
        # 分隔线
        separator = ttk.Separator(control_frame, orient='horizontal')
        separator.pack(fill='x', padx=20, pady=20)
        
        # 结果显示区域
        result_title = tk.Label(
            control_frame,
            text="识别结果",
            font=('Arial', 14, 'bold'),
            bg='#ecf0f1',
            fg='#2c3e50'
        )
        result_title.pack(pady=(0, 10))
        
        # 预测类别
        self.prediction_label = tk.Label(
            control_frame,
            text="等待识别...",
            font=('Arial', 12, 'bold'),
            bg='#ecf0f1',
            fg='#7f8c8d'
        )
        self.prediction_label.pack(pady=5)
        
        # 置信度
        self.confidence_label = tk.Label(
            control_frame,
            text="置信度: ---%",
            font=('Arial', 11),
            bg='#ecf0f1',
            fg='#7f8c8d'
        )
        self.confidence_label.pack(pady=5)
        
        # 置信度进度条
        self.confidence_var = tk.DoubleVar()
        self.confidence_bar = ttk.Progressbar(
            control_frame,
            variable=self.confidence_var,
            maximum=100,
            style='Confidence.Horizontal.TProgressbar'
        )
        self.confidence_bar.pack(fill='x', padx=20, pady=10)
        
        # 详细信息
        detail_title = tk.Label(
            control_frame,
            text="详细信息",
            font=('Arial', 12, 'bold'),
            bg='#ecf0f1',
            fg='#2c3e50'
        )
        detail_title.pack(pady=(10, 5))
        
        # 详细结果文本框
        self.detail_text = tk.Text(
            control_frame,
            height=8,
            width=40,
            font=('Courier', 9),
            bg='white',
            fg='#2c3e50',
            relief='solid',
            bd=1,
            state='disabled'
        )
        self.detail_text.pack(fill='x', padx=15, pady=5)
        
        # 滚动条
        detail_scrollbar = ttk.Scrollbar(control_frame, orient='vertical', command=self.detail_text.yview)
        self.detail_text.configure(yscrollcommand=detail_scrollbar.set)
        
        # 右侧图片显示区域
        image_frame = tk.Frame(main_frame, bg='white', relief='solid', bd=2)
        image_frame.pack(side='right', fill='both', expand=True, padx=(15, 0))
        
        # 图片标题
        image_title = tk.Label(
            image_frame,
            text="图片预览",
            font=('Arial', 16, 'bold'),
            bg='white',
            fg='#2c3e50'
        )
        image_title.pack(pady=(15, 10))
        
        # 创建图片容器框架以便更好地控制布局
        image_container = tk.Frame(image_frame, bg='white')
        image_container.pack(fill='both', expand=True, padx=20, pady=(0, 20))
        
        # 图片显示标签
        self.image_label = tk.Label(
            image_container,
            text="请选择一张图片开始识别",
            font=('Arial', 14),
            bg='white',
            fg='#7f8c8d',
            justify='center'
        )
        self.image_label.pack(fill='both', expand=True)
        
        # 设置进度条样式
        style = ttk.Style()
        style.configure('Processing.Horizontal.TProgressbar', background='#3498db')
        style.configure('Confidence.Horizontal.TProgressbar', background='#27ae60')
        
    def load_model(self):
        """在后台线程中加载模型"""
        def load_in_thread():
            try:
                model_path = 'models/best_pytorch_model.pth'
                
                if not os.path.exists(model_path):
                    self.root.after(0, lambda: self.model_status_label.config(
                        text="[错误] 模型文件不存在",
                        fg='#e74c3c'
                    ))
                    return
                
                # 加载模型
                model = models.resnet18(pretrained=False)
                num_features = model.fc.in_features
                model.fc = nn.Linear(num_features, 2)
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model = model.to(self.device)
                model.eval()
                
                self.model = model
                
                # 更新UI状态
                device_name = "GPU" if self.device.type == 'cuda' else "CPU"
                self.root.after(0, lambda: self.model_status_label.config(
                    text=f"[成功] 模型已加载 ({device_name})",
                    fg='#27ae60'
                ))
                
                # 启用选择按钮
                self.root.after(0, lambda: self.select_btn.config(state='normal'))
                
            except Exception as e:
                error_msg = f"[错误] 模型加载失败: {str(e)}"
                self.root.after(0, lambda: self.model_status_label.config(
                    text=error_msg,
                    fg='#e74c3c'
                ))
                print(f"模型加载错误: {e}")
        
        threading.Thread(target=load_in_thread, daemon=True).start()
    
    def select_image(self):
        """选择图片文件"""
        file_types = [
            ('图片文件', '*.jpg;*.jpeg;*.png;*.bmp;*.gif'),
            ('JPEG文件', '*.jpg;*.jpeg'),
            ('PNG文件', '*.png'),
            ('所有文件', '*.*')
        ]
        
        file_path = filedialog.askopenfilename(
            title="选择要识别的图片",
            filetypes=file_types
        )
        
        if file_path:
            try:
                # 读取图片
                image = Image.open(file_path)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                self.current_image = image
                self.original_image = image.copy()
                
                # 显示图片
                self.display_image()
                
                # 启用识别按钮
                self.classify_btn.config(state='normal')
                
                # 重置结果显示
                self.reset_results()
                
            except Exception as e:
                messagebox.showerror("错误", f"无法加载图片: {str(e)}")
    
    def display_image(self, annotated_image=None):
        """显示图片"""
        try:
            # 使用注释图片或原始图片
            image_to_display = annotated_image if annotated_image else self.current_image
            
            # 获取显示区域实际尺寸
            self.root.update_idletasks()  # 确保布局已完成
            
            # 获取图片容器的实际大小
            container_width = self.image_label.winfo_width()
            container_height = self.image_label.winfo_height()
            
            # 如果容器尺寸为1（还未初始化），使用默认值
            if container_width <= 1 or container_height <= 1:
                max_width = 900
                max_height = 650
            else:
                # 留出一些边距
                max_width = max(container_width - 40, 400)
                max_height = max(container_height - 40, 300)
            
            img_width, img_height = image_to_display.size
            
            # 计算缩放比例，保持纵横比
            scale_w = max_width / img_width
            scale_h = max_height / img_height
            scale = min(scale_w, scale_h, 1.0)  # 不放大图片
            
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            # 调整图片大小
            resized_image = image_to_display.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 转换为Tkinter格式
            photo = ImageTk.PhotoImage(resized_image)
            
            # 更新显示
            self.image_label.config(image=photo, text="", compound='center')
            self.image_label.image = photo  # 保持引用
            
        except Exception as e:
            print(f"显示图片错误: {e}")
    
    def classify_image(self):
        """分类图片"""
        if self.current_image is None or self.model is None:
            return
            
        # 在后台线程中进行推理
        def classify_in_thread():
            try:
                # 更新按钮状态
                self.root.after(0, lambda: self.classify_btn.config(state='disabled', text='识别中...'))
                
                # 预处理图片
                input_tensor = self.transform(self.current_image).unsqueeze(0).to(self.device)
                
                # 推理
                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                    
                    # 获取预测结果
                    predicted_idx = torch.argmax(outputs, dim=1).item()
                    confidence = probabilities[predicted_idx].item() * 100
                    
                    # 类别名称
                    class_names = ['猫', '狗']
                    predicted_class = class_names[predicted_idx]
                    
                    # 各类别概率
                    cat_prob = probabilities[0].item() * 100
                    dog_prob = probabilities[1].item() * 100
                    
                    # 创建带注释的图片
                    annotated_image = self.create_annotated_image(predicted_class, confidence, cat_prob, dog_prob)
                    
                    # 更新UI
                    self.root.after(0, lambda: self.update_results(predicted_class, confidence, cat_prob, dog_prob, annotated_image))
                    
            except Exception as e:
                error_msg = f"识别过程中出错: {str(e)}"
                self.root.after(0, lambda: messagebox.showerror("错误", error_msg))
                print(f"分类错误: {e}")
            finally:
                # 恢复按钮状态
                self.root.after(0, lambda: self.classify_btn.config(state='normal', text='开始识别'))
        
        threading.Thread(target=classify_in_thread, daemon=True).start()
    
    def create_annotated_image(self, predicted_class, confidence, cat_prob, dog_prob):
        """创建带注释的图片"""
        try:
            # 复制原始图片
            annotated = self.original_image.copy()
            draw = ImageDraw.Draw(annotated)
            
            # 获取图片尺寸
            width, height = annotated.size
            
            # 设置字体大小（根据图片大小调整）
            base_font_size = max(16, min(width, height) // 30)
            title_font_size = int(base_font_size * 1.5)
            
            try:
                # 尝试使用系统字体
                title_font = self.get_font(title_font_size)
                text_font = self.get_font(base_font_size)
            except:
                # 如果字体加载失败，使用默认字体
                title_font = None
                text_font = None
            
            # 计算文本位置
            margin = 20
            line_height = base_font_size + 10
            
            # 背景框的坐标
            text_lines = [
                f"预测结果: {predicted_class}",
                f"置信度: {confidence:.1f}%",
                f"猫的概率: {cat_prob:.1f}%",
                f"狗的概率: {dog_prob:.1f}%"
            ]
            
            # 计算背景框大小
            max_text_width = 0
            for line in text_lines:
                if text_font:
                    bbox = draw.textbbox((0, 0), line, font=text_font)
                    text_width = bbox[2] - bbox[0]
                else:
                    text_width = len(line) * base_font_size * 0.6
                max_text_width = max(max_text_width, text_width)
            
            bg_width = max_text_width + margin * 2
            bg_height = len(text_lines) * line_height + margin * 2
            
            # 确定背景框位置（右上角）
            bg_x = width - bg_width - 10
            bg_y = 10
            
            # 绘制半透明背景
            overlay = Image.new('RGBA', annotated.size, (255, 255, 255, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            
            # 绘制背景框
            overlay_draw.rectangle(
                [bg_x, bg_y, bg_x + bg_width, bg_y + bg_height],
                fill=(0, 0, 0, 180),
                outline=(255, 255, 255, 255),
                width=2
            )
            
            # 合并overlay到主图片
            annotated = Image.alpha_composite(annotated.convert('RGBA'), overlay).convert('RGB')
            draw = ImageDraw.Draw(annotated)
            
            # 绘制文本
            y_offset = bg_y + margin
            for i, line in enumerate(text_lines):
                # 根据类别设置颜色
                if i == 0:  # 预测结果行
                    color = '#27ae60' if confidence > 70 else '#f39c12' if confidence > 50 else '#e74c3c'
                elif i == 1:  # 置信度行
                    color = '#3498db'
                else:  # 概率行
                    color = '#ecf0f1'
                
                draw.text(
                    (bg_x + margin, y_offset),
                    line,
                    fill=color,
                    font=text_font
                )
                y_offset += line_height
            
            # 在左上角绘制大标题
            emoji = "[猫]" if predicted_class == "猫" else "[狗]"
            title_text = f"{emoji} {predicted_class}"
            
            # 标题背景
            if title_font:
                title_bbox = draw.textbbox((0, 0), title_text, font=title_font)
                title_width = title_bbox[2] - title_bbox[0]
                title_height = title_bbox[3] - title_bbox[1]
            else:
                title_width = len(title_text) * title_font_size * 0.6
                title_height = title_font_size
            
            title_bg_width = title_width + margin
            title_bg_height = title_height + margin
            
            # 绘制标题背景
            title_overlay = Image.new('RGBA', annotated.size, (255, 255, 255, 0))
            title_overlay_draw = ImageDraw.Draw(title_overlay)
            
            title_overlay_draw.rectangle(
                [10, 10, 10 + title_bg_width, 10 + title_bg_height],
                fill=(52, 73, 94, 200),
                outline=(255, 255, 255, 255),
                width=3
            )
            
            annotated = Image.alpha_composite(annotated.convert('RGBA'), title_overlay).convert('RGB')
            draw = ImageDraw.Draw(annotated)
            
            # 绘制标题文本
            title_color = '#ffffff'
            draw.text(
                (10 + margin//2, 10 + margin//2),
                title_text,
                fill=title_color,
                font=title_font
            )
            
            return annotated
            
        except Exception as e:
            print(f"创建注释图片时出错: {e}")
            return self.original_image
    
    def get_font(self, size):
        """获取字体"""
        try:
            # 尝试加载中文字体
            font_names = [
                "simsun.ttc",  # 宋体
                "simhei.ttf",  # 黑体
                "msyh.ttc",    # 微软雅黑
                "arial.ttf",   # Arial
            ]
            
            for font_name in font_names:
                try:
                    return ImageFont.truetype(font_name, size)
                except:
                    continue
            
            # 如果都失败了，返回默认字体
            return ImageFont.load_default()
        except:
            return ImageFont.load_default()
    
    def update_results(self, predicted_class, confidence, cat_prob, dog_prob, annotated_image):
        """更新结果显示"""
        try:
            # 更新预测类别
            emoji = "[猫]" if predicted_class == "猫" else "[狗]"
            self.prediction_label.config(
                text=f"{emoji} {predicted_class}",
                fg='#27ae60' if confidence > 70 else '#f39c12' if confidence > 50 else '#e74c3c'
            )
            
            # 更新置信度
            self.confidence_label.config(
                text=f"置信度: {confidence:.1f}%"
            )
            self.confidence_var.set(confidence)
            
            # 更新详细信息
            detail_info = f"""
[预测] 预测类别: {predicted_class}
[置信] 置信度: {confidence:.1f}%

[概率分布]
  猫: {cat_prob:.1f}%
  狗: {dog_prob:.1f}%

[设备信息]
  设备: {self.device.type.upper()}
  模型: ResNet18

[结果解释]
{'置信度很高，预测结果可靠' if confidence > 80 else '置信度中等，结果较可靠' if confidence > 60 else '置信度较低，请注意'}
            """.strip()
            
            self.detail_text.config(state='normal')
            self.detail_text.delete('1.0', tk.END)
            self.detail_text.insert('1.0', detail_info)
            self.detail_text.config(state='disabled')
            
            # 显示带注释的图片
            self.display_image(annotated_image)
            
        except Exception as e:
            print(f"更新结果时出错: {e}")
    
    def reset_results(self):
        """重置结果显示"""
        self.prediction_label.config(text="等待识别...", fg='#7f8c8d')
        self.confidence_label.config(text="置信度: ---%")
        self.confidence_var.set(0)
        
        self.detail_text.config(state='normal')
        self.detail_text.delete('1.0', tk.END)
        self.detail_text.config(state='disabled')
    
    def batch_classify(self):
        """批量分类图片"""
        if self.model is None:
            messagebox.showwarning("警告", "请先加载模型")
            return
        
        folder_path = filedialog.askdirectory(title="选择包含图片的文件夹")
        
        if folder_path:
            # 获取所有图片文件
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
            image_files = []
            
            for filename in os.listdir(folder_path):
                if any(filename.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(filename)
            
            if not image_files:
                messagebox.showinfo("信息", "所选文件夹中没有找到图片文件")
                return
            
            # 确认开始批量处理
            result = messagebox.askyesno(
                "确认", 
                f"找到 {len(image_files)} 张图片。\n开始批量识别？"
            )
            
            if result:
                self.start_batch_processing(image_files, folder_path)
    
    def start_batch_processing(self, image_files, folder_path):
        """开始批量处理"""
        # 显示进度条和状态
        self.progress_bar.pack(fill='x', padx=20, pady=5)
        self.batch_status_label.pack(pady=5)
        
        # 禁用按钮
        self.batch_btn.config(state='disabled', text='处理中...')
        self.select_btn.config(state='disabled')
        self.classify_btn.config(state='disabled')
        
        # 在后台线程中处理
        def process_in_thread():
            results = []
            total_files = len(image_files)
            
            try:
                for i, filename in enumerate(image_files):
                    file_path = os.path.join(folder_path, filename)
                    
                    try:
                        # 更新进度
                        progress = (i / total_files) * 100
                        self.root.after(0, lambda p=progress, f=filename: self.update_batch_progress(p, f))
                        
                        # 加载和处理图片
                        image = Image.open(file_path)
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        
                        # 预处理
                        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
                        
                        # 推理
                        with torch.no_grad():
                            outputs = self.model(input_tensor)
                            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                            
                            predicted_idx = torch.argmax(outputs, dim=1).item()
                            confidence = probabilities[predicted_idx].item() * 100
                            
                            class_names = ['猫', '狗']
                            predicted_class = class_names[predicted_idx]
                            
                            cat_prob = probabilities[0].item() * 100
                            dog_prob = probabilities[1].item() * 100
                        
                        # 保存结果
                        results.append({
                            'filename': filename,
                            'predicted_class': predicted_class,
                            'confidence': confidence,
                            'cat_probability': cat_prob,
                            'dog_probability': dog_prob
                        })
                        
                    except Exception as e:
                        print(f"处理文件 {filename} 时出错: {e}")
                        results.append({
                            'filename': filename,
                            'predicted_class': '错误',
                            'confidence': 0,
                            'cat_probability': 0,
                            'dog_probability': 0,
                            'error': str(e)
                        })
                
                # 完成处理
                self.root.after(0, lambda: self.finish_batch_processing(results, folder_path))
                
            except Exception as e:
                error_msg = f"批量处理过程中出错: {str(e)}"
                self.root.after(0, lambda: self.handle_batch_error(error_msg))
        
        threading.Thread(target=process_in_thread, daemon=True).start()
    
    def update_batch_progress(self, progress, current_file):
        """更新批量处理进度"""
        self.progress_var.set(progress)
        self.batch_status_label.config(text=f"处理中: {current_file}")
    
    def finish_batch_processing(self, results, folder_path):
        """完成批量处理"""
        try:
            # 保存结果到CSV文件
            import pandas as pd
            from datetime import datetime
            
            # 创建DataFrame
            df = pd.DataFrame(results)
            
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"批量识别结果_{timestamp}.csv"
            csv_path = os.path.join('data', csv_filename)
            
            # 确保data目录存在
            os.makedirs('data', exist_ok=True)
            
            # 保存CSV
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            
            # 统计结果
            total_count = len(results)
            cat_count = len([r for r in results if r['predicted_class'] == '猫'])
            dog_count = len([r for r in results if r['predicted_class'] == '狗'])
            error_count = len([r for r in results if r['predicted_class'] == '错误'])
            
            # 显示结果摘要
            summary = f"""
[统计] 处理统计:
  总计: {total_count} 张图片
  识别成功: {total_count - error_count} 张
  处理失败: {error_count} 张

[结果] 识别结果:
  猫: {cat_count} 张
  狗: {dog_count} 张

[保存] 详细结果已保存至:
{csv_path}
            """.strip()
            
            self.detail_text.config(state='normal')
            self.detail_text.delete('1.0', tk.END)
            self.detail_text.insert('1.0', summary)
            self.detail_text.config(state='disabled')
            
            messagebox.showinfo(
                "批量处理完成", 
                f"成功处理 {total_count} 张图片\n"
                f"猫: {cat_count} 张，狗: {dog_count} 张\n"
                f"结果已保存至: {csv_filename}"
            )
            
        except Exception as e:
            error_msg = f"保存结果时出错: {str(e)}"
            self.handle_batch_error(error_msg)
        finally:
            self.restore_ui_after_batch()
    
    def handle_batch_error(self, error_msg):
        """处理批量处理错误"""
        messagebox.showerror("批量处理错误", error_msg)
        self.restore_ui_after_batch()
    
    def restore_ui_after_batch(self):
        """批量处理后恢复UI状态"""
        # 隐藏进度条
        self.progress_bar.pack_forget()
        self.batch_status_label.pack_forget()
        
        # 恢复按钮状态
        self.batch_btn.config(state='normal', text='批量识别')
        self.select_btn.config(state='normal')
        if self.current_image is not None:
            self.classify_btn.config(state='normal')
        
        # 重置进度
        self.progress_var.set(0)
        self.batch_status_label.config(text="")

def main():
    root = tk.Tk()
    app = CatDogClassifierUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 