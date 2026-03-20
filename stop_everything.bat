@echo off
taskkill /FI "WINDOWTITLE eq MedAI Standalone" /T /F
taskkill /FI "WINDOWTITLE eq MedAI Frontend" /T /F
taskkill /FI "WINDOWTITLE eq MedAI Backend" /T /F
