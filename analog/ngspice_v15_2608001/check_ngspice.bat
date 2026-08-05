@echo off
pushd "%~dp0"
py ngspice_runner.py
echo.
echo runner_probe.log location:
echo %~dp0results\runner_probe.log
pause
popd
