@echo off
chcp 65001 >nul
title 猫狗分类器 - AI图像识别

echo ================================================================
echo                    猫狗分类器 - AI图像识别
echo ================================================================
echo.

:: 检查虚拟环境是否存在
if not exist "venv_py312\Scripts\python.exe" (
    echo [错误] 虚拟环境不存在！
    echo 请确保 venv_py312 目录存在且包含正确的Python环境
    echo.
    pause
    exit /b 1
)

echo [信息] 检测到虚拟环境：venv_py312
echo [信息] 正在启动猫狗分类器界面...
echo.

:: 使用虚拟环境的Python运行GUI
"venv_py312\Scripts\python.exe" image_classifier_ui.py

:: 检查运行结果
if %ERRORLEVEL% neq 0 (
    echo.
    echo [错误] 程序运行失败，错误代码: %ERRORLEVEL%
    echo.
    echo 可能的解决方案:
    echo 1. 确保虚拟环境已正确安装所有依赖
    echo 2. 检查模型文件是否存在: models/best_pytorch_model.pth
    echo 3. 运行 python run_project.py 检查项目状态
    echo.
    pause
) else (
    echo.
    echo [完成] 程序已正常退出
)

pause 