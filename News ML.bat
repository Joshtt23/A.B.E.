@echo off

REM Check if running as administrator
NET SESSION >nul 2>&1
if %errorLevel% == 0 (
    echo Running as administrator...
) else (
    echo Requesting admin privileges...
    echo Please run this script as administrator.
    powershell -Command "Add-Type -TypeDefinition 'using System;class P{static void Main(){Console.WriteLine(\"Press any key to exit...\");Console.ReadKey();}}' -OutputAssembly '%temp%\admincheck.exe'"
    "%temp%\admincheck.exe"
    exit
)

REM Set the correct working directory
set "WORKING_DIR=%~dp0"

REM Activate the virtual environment if present
if exist "%WORKING_DIR%\venv\Scripts\activate.bat" (
    call "%WORKING_DIR%\venv\Scripts\activate.bat"
)

REM Run the Python script
python "%WORKING_DIR%\run.py"

REM Deactivate the virtual environment if present
if exist "%WORKING_DIR%\venv\Scripts\deactivate.bat" (
    call "%WORKING_DIR%\venv\Scripts\deactivate.bat"
)

pause
