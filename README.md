# MedAI Portfolio Workspace

This workspace contains two layers:

- `project/`:
  the main application with backend, login page, dashboard, datasets, and trained artifacts
- `MedAI_Portfolio_Dashboard.html`:
  a standalone prompt-based dashboard for demo and portfolio presentation

## Recommended Entry Points

### Main app

Run:

```powershell
cd "d:\minor project 6 sem\project\frontend"
py -m http.server 8080
```

Open:

```text
http://127.0.0.1:8080
```

### Backend

Run:

```powershell
cd "d:\minor project 6 sem\project\backend"
py app.py
```

Open:

```text
http://127.0.0.1:5000
```

### Standalone dashboard

Run:

```powershell
cd "d:\minor project 6 sem"
py -m http.server 8092
```

Open:

```text
http://127.0.0.1:8092/MedAI_Portfolio_Dashboard.html
```

## One-Click Launchers

- `run_backend.bat`
- `run_main_frontend.bat`
- `run_standalone_dashboard.bat`
- `run_everything.bat`

For full project details, see:

- `project/README.md`
