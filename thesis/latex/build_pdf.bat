@echo off
setlocal
cd /d "%~dp0"

set JOB=physics_informed_bearing_rul_thesis
set "MIKTEX_BIN=%LOCALAPPDATA%\Programs\MiKTeX\miktex\bin\x64"
set "PDFLATEX=pdflatex"
set "BIBTEX=bibtex"

where pdflatex >nul 2>nul
if errorlevel 1 (
    if exist "%MIKTEX_BIN%\pdflatex.exe" (
        set "PDFLATEX=%MIKTEX_BIN%\pdflatex.exe"
    ) else (
        echo pdflatex was not found. Make sure MiKTeX is installed and on PATH.
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

"%PDFLATEX%" -interaction=nonstopmode -halt-on-error -jobname=%JOB% main.tex
if errorlevel 1 goto fail

"%BIBTEX%" %JOB%
if errorlevel 1 goto fail

"%PDFLATEX%" -interaction=nonstopmode -halt-on-error -jobname=%JOB% main.tex
if errorlevel 1 goto fail

"%PDFLATEX%" -interaction=nonstopmode -halt-on-error -jobname=%JOB% main.tex
if errorlevel 1 goto fail

echo.
echo Built %CD%\%JOB%.pdf
pause
exit /b 0

:fail
echo.
echo LaTeX build failed. Check %JOB%.log for details.
pause
exit /b 1
