@echo off
echo ============================================
echo Installing UMAP and SHAP packages
echo ============================================
echo.

REM Try the main Python installation
echo [1/3] Installing in Python 3.12...
C:\Users\Silver\AppData\Local\Programs\Python\Python312\python.exe -m pip install umap-learn shap
echo.

REM Try python command (might be different environment)
echo [2/3] Installing via 'python' command...
python -m pip install umap-learn shap
echo.

REM Try python3 command
echo [3/3] Installing via 'python3' command...
python3 -m pip install umap-learn shap 2>nul
echo.

echo ============================================
echo Installation complete!
echo ============================================
echo.
echo Now restart VS Code kernel:
echo   1. In VS Code, click "Restart" button in notebook toolbar
echo   2. Or press Ctrl+Shift+P and select "Notebook: Restart Kernel"
echo.
pause
