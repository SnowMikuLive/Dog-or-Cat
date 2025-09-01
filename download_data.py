#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载Kaggle猫狗数据集
"""

import os
import zipfile
from pathlib import Path
import kaggle

def download_dogs_cats_data():
    """下载Dogs vs Cats数据集"""
    
    # 创建数据目录
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    print("[下载] 开始下载Dogs vs Cats数据集...")
    
    try:
        # 下载数据集
        kaggle.api.competition_download_files(
            'dogs-vs-cats-redux-kernels-edition',
            path=str(data_dir),
            quiet=False
        )
        
        print("[OK] 数据集下载完成")
        
        # 解压文件
        print("[解压] 开始解压数据集...")
        
        zip_files = list(data_dir.glob('*.zip'))
        for zip_file in zip_files:
            print(f"解压: {zip_file}")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            
            # 删除zip文件
            zip_file.unlink()
        
        print("[OK] 数据集解压完成")
        
        # 显示数据结构
        print("\n[目录] 数据目录结构:")
        for item in data_dir.rglob('*'):
            if item.is_file():
                print(f"  {item}")
        
    except Exception as e:
        print(f"[错误] 数据下载失败: {e}")
        print("请检查网络连接和Kaggle API配置")

if __name__ == "__main__":
    download_dogs_cats_data() 