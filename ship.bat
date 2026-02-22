@echo off
set /p msg="Enter your commit message: "

echo --- 1. Pulling latest from GitHub (Jules' fixes) ---
git pull origin main

echo --- 2. Staging your local changes ---
git add .

echo --- 3. Committing with message: %msg% ---
git commit -m "%msg%"

echo --- 4. Pushing to GitHub ---
git push origin main

echo --- Done! Your code is now synced and pushed. ---
pause