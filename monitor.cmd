@echo off
REM Kestrel training monitor. Optional arg: run directory (default experiments\nano_d).
set RUNDIR=%~1
if "%RUNDIR%"=="" set RUNDIR=%~dp0experiments\nano_d
start "" "%~dp0tools\KestrelMonitor\bin\Release\net8.0-windows\KestrelMonitor.exe" "%RUNDIR%"
