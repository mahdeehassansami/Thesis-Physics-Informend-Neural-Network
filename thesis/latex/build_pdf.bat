@echo off
setlocal
cd /d "%~dp0"

set JOB=physics_informed_bearing_rul_thesis
set "MIKTEX_BIN=%LOCALAPPDATA%\Programs\MiKTeX\miktex\bin\x64"
set "LATEX=xelatex"
set "BIBTEX=bibtex"

where xelatex >nul 2>nul
if errorlevel 1 (
    if exist "%MIKTEX_BIN%\xelatex.exe" (
        set "LATEX=%MIKTEX_BIN%\xelatex.exe"
    ) else (
        echo xelatex was not found. Make sure MiKTeX is installed and on PATH.
        pause
        exit /b 1
    )
)

where bibtex >nul 2>nul
if errorlevel 1 (
    if exist "%MIKTEX_BIN%\bibtex.exe" (
        set "BIBTEX=%MIKTEX_BIN%\bibtex.exe"
    ) else (
        echo bibtex was not found. Make sure MiKTeX is installed and on PATH.
        pause
        exit /b 1
    )
)

"%LATEX%" -interaction=nonstopmode -halt-on-error -jobname=%JOB% main.tex
if errorlevel 1 goto fail

"%BIBTEX%" %JOB%
if errorlevel 1 goto fail

"%LATEX%" -interaction=nonstopmode -halt-on-error -jobname=%JOB% main.tex
if errorlevel 1 goto fail

"%LATEX%" -interaction=nonstopmode -halt-on-error -jobname=%JOB% main.tex
if errorlevel 1 goto fail

echo.
copy /Y "%JOB%.pdf" "main.pdf" >nul
echo Built %CD%\%JOB%.pdf
echo Synced %CD%\main.pdf
pause
exit /b 0

:fail
echo.
echo LaTeX build failed. Check %JOB%.log for details.
pause
exit /b 1
