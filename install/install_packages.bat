@echo off

set packagelist=packages.txt

set packagelist=%~dp0%packagelist%
set packagelist=%packagelist:\=/%


"C:\Python_3.10\python.exe" -m pip install -r "%packagelist%"

echo|set /p="Press any key to exit..."
pause >nul 2>&1