@echo off
pushd "%~dp0"
if not exist "results" mkdir "results"
>>"results\launcher.log" echo.
>>"results\launcher.log" echo [%date% %time%] GUI START
>>"results\launcher.log" echo app_dir=%CD%
>>"results\launcher.log" where py 2>&1
>>"results\launcher.log" where ngspice 2>&1
>>"results\launcher.log" where ngspice_con 2>&1
py ngspice_channel_sweeper.py >>"results\launcher.log" 2>&1
set "APP_EXIT_CODE=%ERRORLEVEL%"
>>"results\launcher.log" echo [%date% %time%] GUI END returncode=%APP_EXIT_CODE%
if not "%APP_EXIT_CODE%"=="0" pause
popd
exit /b %APP_EXIT_CODE%
