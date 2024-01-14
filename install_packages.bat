@echo off

set PyPath=C:\Python_3.10\python.exe

echo:
echo Installing numpy ^(1^.24^.3^)^:
"%PyPath%" -m pip install numpy==1.24.3

echo:
echo Installing gymnasium ^(0^.28^.1^)^:
"%PyPath%" -m pip install gymnasium==0.28.1

echo:
echo Installing mujoco ^(2^.3^.7^)^:
"%PyPath%" -m pip install mujoco==2.3.7

echo:
echo Installing wandb ^(0^.15^.0^)^:
"%PyPath%" -m pip install wandb==0.15.0

echo:
echo Installing imageio^:
"%PyPath%" -m pip install imageio

echo:
echo Installing torch ^(2^.1^.2^) with CUDA support^:
"%PyPath%" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121


echo|set /p="Press any key to exit..."
pause >nul 2>&1
