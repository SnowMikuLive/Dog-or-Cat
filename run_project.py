#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
猫狗分类项目主启动脚本
"""

import subprocess
import sys
import os
from pathlib import Path

def get_python_command():
    """获取正确的 Python 命令（优先使用虚拟环境）"""
    venv_python = os.path.join(os.path.dirname(__file__), 'venv_py312', 'Scripts', 'python.exe')
    if os.path.exists(venv_python):
        return venv_python
    return 'python'

def run_command(command, description):
    """运行命令并显示进度"""
    print(f"\n{'='*60}")
    print(f"[执行] {description}")
    print(f"{'='*60}")
    
    # 如果是 Python 命令，自动使用虚拟环境的 Python
    if command.startswith('python '):
        python_exe = get_python_command()
        command = command.replace('python ', f'"{python_exe}" ', 1)
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"[OK] {description} 完成")
            if result.stdout:
                print(result.stdout)
        else:
            print(f"[错误] {description} 失败")
            print(f"错误信息: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"[错误] 执行命令时出错: {e}")
        return False
    
    return True

def check_dependencies():
    """检查Python依赖"""
    print("[检查] 检查Python依赖...")
    
    # 使用虚拟环境的Python来检查依赖
    python_exe = get_python_command()
    
    required_packages = [
        'torch', 'numpy', 'pandas', 'matplotlib', 
        'seaborn', 'PIL', 'sklearn', 'kaggle'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            # 使用虚拟环境的Python检查包
            result = subprocess.run(
                f'"{python_exe}" -c "import {package}"', 
                shell=True, 
                capture_output=True, 
                text=True
            )
            if result.returncode != 0:
                missing_packages.append(package)
        except Exception:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"[缺少] 缺少以下依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    else:
        print("[OK] 所有依赖包已安装")
        return True

def main():
    """主函数"""
    print("[猫狗分类] 欢迎使用猫狗分类项目！")
    print("本项目将引导您完成从数据下载到模型训练的完整流程")
    
    while True:
        print("\n" + "="*60)
        print("请选择要执行的步骤:")
        print("1. 安装依赖 (pip install -r requirements.txt)")
        print("2. 配置Kaggle API")
        print("3. 下载数据集")
        print("4. 数据预处理")
        print("5. 训练模型 (PyTorch) - 推荐")
        print("5t. 训练模型 (TensorFlow) - 可能有兼容性问题")
        print("6. 评估模型")
        print("7. 使用模型预测")
        print("8. 运行完整PyTorch流程 (步骤2-6)")
        print("9. 检查项目状态")
        print("0. 退出")
        print("="*60)
        
        choice = input("请输入选择 (0-9): ").strip()
        
        if choice == '0':
            print("[再见] 再见！")
            break
        
        elif choice == '1':
            # 安装依赖
            run_command("pip install -r requirements.txt", "安装Python依赖包")
        
        elif choice == '2':
            # 配置Kaggle API
            run_command("python setup_kaggle.py", "配置Kaggle API")
        
        elif choice == '3':
            # 下载数据
            run_command("python download_data.py", "下载猫狗数据集")
        
        elif choice == '4':
            # 数据预处理
            run_command("python data_preprocessing.py", "数据预处理")
        
        elif choice == '5':
            # 训练模型 (PyTorch)
            print("\n注意: PyTorch模型训练可能需要较长时间（30分钟到几小时），取决于您的硬件配置")
            print("PyTorch对Python 3.13支持更好，推荐使用")
            print("支持CUDA加速训练")
            confirm = input("是否继续? (y/N): ").strip().lower()
            if confirm in ['y', 'yes']:
                run_command("python train_model_pytorch.py", "训练PyTorch猫狗分类模型")
        
        elif choice == '5t':
            # 训练模型 (TensorFlow)
            print("\n注意: TensorFlow模型训练可能需要较长时间（30分钟到几小时），取决于您的硬件配置")
            print("注意: TensorFlow可能不支持Python 3.13，建议使用PyTorch (选项5)")
            confirm = input("是否继续? (y/N): ").strip().lower()
            if confirm in ['y', 'yes']:
                run_command("python train_model.py", "训练TensorFlow猫狗分类模型")
        
        elif choice == '6':
            # 评估模型
            run_command("python evaluate_model.py", "评估模型性能")
        
        elif choice == '7':
            # 预测
            print("\n选择预测模式:")
            print("a. 单张图片预测")
            print("b. 批量图片预测")
            print("c. 测试数据集预测 (Kaggle格式)")
            
            pred_choice = input("请选择 (a/b/c): ").strip().lower()
            
            if pred_choice == 'a':
                img_path = input("请输入图片路径: ").strip()
                if img_path and Path(img_path).exists():
                    run_command(f'python predict.py --image "{img_path}"', "单张图片预测")
                else:
                    print("[错误] 图片路径无效")
            
            elif pred_choice == 'b':
                img_dir = input("请输入图片目录路径: ").strip()
                if img_dir and Path(img_dir).exists():
                    output_file = f"predictions_{Path(img_dir).name}.csv"
                    run_command(f'python predict.py --batch "{img_dir}" --output "{output_file}"', "批量图片预测")
                else:
                    print("[错误] 目录路径无效")
            
            elif pred_choice == 'c':
                print("将预测测试数据集并生成Kaggle提交文件...")
                run_command('python predict.py --test --output "kaggle_submission.csv"', "测试数据集预测")
        
        elif choice == '8':
            # 运行完整PyTorch流程
            print("\n开始运行完整PyTorch流程...")
            print("使用PyTorch进行训练，支持CUDA加速")
            
            steps = [
                ("python setup_kaggle.py", "配置Kaggle API"),
                ("python download_data.py", "下载数据集"),
                ("python data_preprocessing.py", "数据预处理"),
            ]
            
            success = True
            for command, description in steps:
                if not run_command(command, description):
                    success = False
                    break
            
            if success:
                print("\n数据准备完成，现在开始训练PyTorch模型...")
                print("支持CUDA加速，模型训练可能需要较长时间，请耐心等待")
                confirm = input("是否开始PyTorch训练? (y/N): ").strip().lower()
                
                if confirm in ['y', 'yes']:
                    run_command("python train_model_pytorch.py", "训练PyTorch模型")
                    run_command("python evaluate_model.py", "评估PyTorch模型")
                    print("完整PyTorch流程执行完成！")
        
        elif choice == '9':
            # 检查项目状态
            print("\n项目状态检查:")
            
            # 检查文件存在情况
            files_to_check = [
                ("requirements.txt", "依赖文件"),
                ("setup_kaggle.py", "Kaggle配置脚本"),
                ("download_data.py", "数据下载脚本"),
                ("data_preprocessing.py", "数据预处理脚本"),
                ("train_model.py", "模型训练脚本"),
                ("evaluate_model.py", "模型评估脚本"),
                ("predict.py", "预测脚本"),
            ]
            
            for filename, description in files_to_check:
                status = "存在" if Path(filename).exists() else "不存在"
                print(f"{description}: {filename}")
            
            # 检查目录
            dirs_to_check = [
                ("data", "数据目录"),
                ("data/organized", "整理后数据目录"),
                ("models", "模型目录"),
            ]
            
            print("\n[目录] 目录状态:")
            for dirname, description in dirs_to_check:
                if Path(dirname).exists():
                    if dirname == "data":
                        files_count = len(list(Path(dirname).rglob("*")))
                        print(f"[存在] {description}: {dirname} ({files_count} 个文件)")
                    elif dirname == "models":
                        model_files = list(Path(dirname).glob("*.h5"))
                        print(f"[存在] {description}: {dirname} ({len(model_files)} 个模型文件)")
                    else:
                        print(f"[存在] {description}: {dirname}")
                else:
                    print(f"[缺失] {description}: {dirname}")
            
            # 检查依赖
            print(f"\n[检查] 依赖检查:")
            check_dependencies()
        
        else:
            print("无效选择，请重新输入")

if __name__ == "__main__":
    main() 