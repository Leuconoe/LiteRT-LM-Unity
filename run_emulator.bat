@echo off
set "EMULATOR_PATH=%LOCALAPPDATA%\Android\Sdk\emulator\emulator.exe"
set "AVD_NAME=Medium_Phone_API_36.0"

if not exist "%EMULATOR_PATH%" (
    echo [ERROR] Emulator not found at: %EMULATOR_PATH%
    echo Please check your Android SDK installation path.
    pause
    exit /b 1
)

echo Starting AVD: %AVD_NAME%...
start "" "%EMULATOR_PATH%" -avd %AVD_NAME%
echo Emulator is starting in the background.
exit /b 0
