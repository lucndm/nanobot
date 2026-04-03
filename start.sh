#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

cd /data/data/com.termux/files/home/workspaces/github.com/nanobot

echo '>> Pulling nanobot...'
git fetch origin
git reset --hard origin/develop

echo '>> Pulling ws_nanobot...'
cd ~/.nanobot/workspace
git stash --include-untracked --quiet 2>/dev/null || true
git fetch origin
git reset --hard origin/main
git stash pop --quiet 2>/dev/null || true

echo '>> Stopping nanobot...'
pkill -f 'nanobot gateway' 2>/dev/null || true
sleep 2

echo '>> Starting nanobot...'
source ~/.bashrc
nanobot gateway &disown
sleep 3

if pgrep -f 'nanobot gateway' > /dev/null; then
    echo '>> nanobot started OK'
else
    echo '>> ERROR: nanobot failed to start'
    exit 1
fi
