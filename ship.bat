@echo off
set /p msg="Enter your commit message: "

echo.
echo --- 1. Pulling latest from GitHub ---
git pull origin main
if %errorlevel% neq 0 goto error

echo.
echo --- 2. Staging your local changes ---
git add .

echo.
echo --- 3. Committing changes ---
:: We let commit run, but we don't check %errorlevel% immediately after.
:: This way, if there is "nothing to commit", the script continues to the push.
git commit -m "%msg%"

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
echo [ERROR] Something went wrong (likely a Merge Conflict during pull).
echo Check the output above.
pause
exit /b