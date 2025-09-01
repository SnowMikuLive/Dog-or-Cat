#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查CUDA支持状态
"""

import torch

def check_cuda():
    print("=" * 60)
    print("PyTorch CUDA 支持检查")
    print("=" * 60)
    
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"CUDA设备数量: {torch.cuda.device_count()}")
        print(f"当前CUDA设备: {torch.cuda.current_device()}")
        print(f"设备名称: {torch.cuda.get_device_name()}")
        
        device_props = torch.cuda.get_device_properties(0)
        print(f"显存总量: {device_props.total_memory / 1024**3:.1f} GB")
        print(f"计算能力: {device_props.major}.{device_props.minor}")
        
        # 测试CUDA操作
        try:
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.matmul(x, y)
            print(f"[成功] GPU矩阵运算测试通过")
            print(f"结果设备: {z.device}")
        except Exception as e:
            print(f"[失败] GPU测试失败: {e}")
    else:
        print("CUDA不可用，将使用CPU")
        # 检查PyTorch编译信息
        print(f"PyTorch编译信息: {torch.__config__.show()}")

if __name__ == "__main__":
    check_cuda() 