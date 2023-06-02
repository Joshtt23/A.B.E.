@echo off

REM Check if running as administrator
NET SESSION >nul 2>&1
if %errorLevel% == 0 (
    echo Running as administrator...
) else (
    echo This script requires administrator privileges.
    echo Please run this script as administrator.
    echo Press any key to exit...
    pause >nul
    exit
)

REM Set the correct working directory
set "WORKING_DIR=%~dp0"

REM Activate the virtual environment if present
if exist "%WORKING_DIR%\venv\Scripts\activate.bat" (
    call "%WORKING_DIR%\venv\Scripts\activate.bat"
)

REM Run the Python script
python "%WORKING_DIR%\run-server.py"

REM Deactivate the virtual environment if present
if exist "%WORKING_DIR%\venv\Scripts\deactivate.bat" (
    call "%WORKING_DIR%\venv\Scripts\deactivate.bat"
)

exit
