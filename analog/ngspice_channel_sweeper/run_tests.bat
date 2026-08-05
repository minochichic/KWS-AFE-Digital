@echo off
pushd "%~dp0"
set "MPLCONFIGDIR=%TEMP%\ngspice_channel_sweeper_mpl"
if not exist "%MPLCONFIGDIR%" mkdir "%MPLCONFIGDIR%"
py -m unittest discover -s tests -v
set "APP_EXIT_CODE=%ERRORLEVEL%"
if not "%APP_EXIT_CODE%"=="0" pause
popd
exit /b %APP_EXIT_CODE%
