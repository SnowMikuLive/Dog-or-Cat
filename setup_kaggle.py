#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kaggle API配置脚本
"""

import os
import json
from pathlib import Path

def setup_kaggle_api():
    """设置Kaggle API凭证"""
    
    # Kaggle API凭证
    kaggle_credentials = {
        "username": "snowmikulive",
        "key": "a4bc438e190eb1f41dcc7bab22cbc2c6"
    }
    
    # 创建.kaggle目录
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_dir.mkdir(exist_ok=True)
    
    # 写入kaggle.json文件
    kaggle_json_path = kaggle_dir / 'kaggle.json'
    with open(kaggle_json_path, 'w') as f:
        json.dump(kaggle_credentials, f)
    
    # 设置文件权限（仅限Unix系统）
    if os.name != 'nt':  # 非Windows系统
        os.chmod(kaggle_json_path, 0o600)
    
    print(f"[OK] Kaggle API凭证已配置到: {kaggle_json_path}")
    print("[OK] 现在可以使用Kaggle API下载数据集了")

if __name__ == "__main__":
    setup_kaggle_api() 