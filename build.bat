@echo off
REM One-click local build. Needs Python 3.10+ and ffmpeg.exe + ffprobe.exe on PATH.
pip install -r requirements.txt pyinstaller || exit /b 1
copy /y "%~dp0..\ffmpeg.exe" . >nul 2>&1
for %%F in (ffmpeg.exe ffprobe.exe) do if not exist %%F for /f "delims=" %%P in ('where %%F') do copy /y "%%P" . >nul
pyinstaller --noconfirm --windowed --name HinglishCaptioner --collect-all faster_whisper --collect-all ctranslate2 --collect-all indic_transliteration --add-binary "ffmpeg.exe;." --add-binary "ffprobe.exe;." app.py
echo.
echo Done. Your app is in dist\HinglishCaptioner\HinglishCaptioner.exe
pause
