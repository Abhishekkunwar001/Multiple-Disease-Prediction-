@echo off
start "MedAI Standalone" cmd /k "cd /d d:\minor project 6 sem && py -m http.server 8092"
start "MedAI Frontend" cmd /k "cd /d d:\minor project 6 sem\project\frontend && py -m http.server 8080"
start "MedAI Backend" cmd /k "cd /d d:\minor project 6 sem\project\backend && py app.py"
