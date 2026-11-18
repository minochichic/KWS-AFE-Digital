@echo off
rem 발표용 두 화면 (영상 촬영). afe_view.cmd 로 뷰어(:8765)를 먼저 띄운 뒤 실행한다.
rem   화면 1 http://127.0.0.1:8780/    현재 답 + 최근 캡처 10개
rem   화면 2 http://127.0.0.1:8780/cm  정답 x FPGA 답 혼동행렬
rem   afe_demo.cmd --refs out\afe\grid_leak2   옵션은 그대로 afe_demo 에 넘긴다
chcp 65001 >nul
cd /d "%~dp0..\..\.."
set PY=%KWS_PY%
if "%PY%"=="" if exist "%USERPROFILE%\kws_venv\Scripts\python.exe" set PY=%USERPROFILE%\kws_venv\Scripts\python.exe
if "%PY%"=="" set PY=python
"%PY%" -m experiments.afe_demo --open %*
pause
