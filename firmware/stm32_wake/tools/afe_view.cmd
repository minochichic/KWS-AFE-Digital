@echo off
rem AFE 실시간 보기: 서버(SWD 로 보드 읽기)를 띄우고 브라우저로 http://127.0.0.1:8765 를 연다.
rem 이 창을 닫으면 서버가 멈춘다. 보드는 -D WAKE_AFELOG 빌드여야 한다.
rem   afe_view.cmd                          out\afe\*\ref.json 전부 -- 페이지에서 재생 중인 ARB 를 고른다
rem   afe_view.cmd out\afe\<clip>\ref.json  그 클립 하나만 기준으로
chcp 65001 >nul
cd /d "%~dp0..\..\.."
set PY=%KWS_PY%
if "%PY%"=="" if exist "%USERPROFILE%\kws_venv\Scripts\python.exe" set PY=%USERPROFILE%\kws_venv\Scripts\python.exe
if "%PY%"=="" set PY=python
if "%~1"=="" (set REFARG=) else (set REFARG=--ref "%~1")
"%PY%" -m experiments.afe_web %REFARG% --open
pause
