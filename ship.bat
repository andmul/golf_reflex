@echo off
set /p msg="Enter your commit message: "

echo.
echo --- 1. Pulling latest from GitHub ---
git pull origin main
if %errorlevel% neq 0 goto error

echo.
echo --- 2. Staging your local changes ---
git add .
if %errorlevel% neq 0 goto error

echo.
echo --- 3. Committing with message: %msg% ---
git commit -m "%msg%"
if %errorlevel% neq 0 goto error

echo.
echo --- 4. Pushing to GitHub ---
git push origin main
if %errorlevel% neq 0 goto error

echo.
echo --- Done! Your code is now synced and pushed. ---
pause
exit /b

:error
echo.
echo [ERROR] Something went wrong! The script has stopped to protect your Git history.
echo Please scroll up to see what failed (e.g., a merge conflict), fix it, and try again.
pause
exit /b